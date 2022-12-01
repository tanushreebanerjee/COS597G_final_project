# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import string
import numpy as np
import pickle as pkl
import math
import torch

from collections import defaultdict
from functools import partial
from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

class GPT2Data(object):

    def __init__(self, logger=None, tokenizer=None, use_demonstrations=True, k=16,
                 max_length=1024, max_length_per_example=256,
                 do_tensorize=False, tensorize_dir=None, n_process=None, n_gpu=None, local_rank=-1):

        self.logger = logger
        self.tokenizer = tokenizer
        self.use_demonstrations = use_demonstrations
        self.k = k
        self.max_length = max_length
        self.max_length_per_example = max_length_per_example

        self.do_tensorize = do_tensorize
        self.tensorize_dir = tensorize_dir
        self.n_process = n_process
        self.n_gpu = n_gpu
        self.local_rank = local_rank

        self.tensorized_inputs = None
        self.metadata = None

        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __len__(self):
        if self.tensorized_inputs is None:
            return 0
        return len(self.tensorized_inputs["input_ids"])

    def __str__(self):
        text = "[GPT2Data]: "
        if self.use_demonstrations:
            text += "%d demonstrations\n" % self.k
        else:
            text += "no demonstrations\n"
        if self.metadata is None:
            text += "Currently not containing any examples"
        else:
            text += "Currently containing %d examples with %d tensors to be fed in\n" % (len(self.metadata), len(self))
            text += "\n"
            text += self.print_tensorized_example(return_string=True)
        return ("="*50) + "\n" + text + "\n" + ("="*50)

    def get_dataloader(self, batch_size, is_training):
        inputs = self.tensorized_inputs
        for k, v in inputs.items():
            if type(v)==list:
                inputs[k] = torch.LongTensor(v)
        shape = inputs["input_ids"].shape
        self.logger.info(shape)
        for v in inputs.values():
            assert v.shape==shape
        dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
        if is_training:
            sampler=RandomSampler(dataset)
        else:
            sampler=SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
        return dataloader

    def print_tensorized_example(self, return_string=False):
        assert self.tensorized_inputs is not None

        idx = 0
        text = "Checking the first example..."
        input_ids = self.tensorized_inputs["input_ids"][idx]
        attention_mask = self.tensorized_inputs["attention_mask"][idx]
        if type(input_ids)!=list:
            input_ids = input_ids.numpy().tolist()
        if type(attention_mask)!=list:
            attention_mask = attention_mask.numpy().tolist()

        text += "\nInput:\n"
        if attention_mask[-1] == 1:
            text += self.tokenizer.decode(input_ids)
        else:
            text += self.tokenizer.decode(input_ids[:attention_mask.index(0)])

        if return_string:
            return text

        if self.local_rank<=0:
            self.logger.info(text)

    def _prepro_each_datapoint(self, dp, is_first=True, for_demonstrations=False, add_newlines=True):
        dp = dp.copy()
        assert type(dp["output"]) == list

        ## GPT-J
        if add_newlines:
            dp["output"] = dp["output"][0]

            ## If first demonstration
            ## separate the output of this datapoint
            ## from the input of this datapoint
            if is_first:
                dp["output"] = "\n" + dp["output"]

            ## If second demonstration or later, or if test datapoint
            ## separate the input of this datapoint
            ## from the output of the previous datapoint
            else:
                dp["input"] = "\n\n\n" + dp["input"]
                dp["output"] = "\n" + dp["output"]

        ## GPT-2
        else:

            if "context" in dp:
                dp["input"] = "Context: " + dp["context"] + "\n" + "Question: " + dp["input"]
            else:
                dp["input"] = "Question: " + dp["input"] + "\n"

            if for_demonstrations:
                dp["output"] = "Answer: " + dp["output"][0] + "\n"
            else:
                dp["output"] = "Answer: "

        input_tokens = self.tokenizer(dp["input"])["input_ids"]
        output_tokens = self.tokenizer(dp["output"])["input_ids"]

        ## processing demonstrations
        if for_demonstrations:

            ## cut off some input if necessary
            if len(input_tokens)>=self.max_length_per_example - 2 - len(output_tokens):
                input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, (len(input_tokens), len(output_tokens), self.max_length_per_example)

        ## processing test datapoints
        else:

            ## cut off some input if necessary
            if len(input_tokens)>=self.max_length_per_example - 2:
                input_tokens = input_tokens[:self.max_length_per_example - 2]

            assert len(input_tokens)+2<=self.max_length_per_example, (len(input_tokens), self.max_length_per_example)

        return input_tokens, output_tokens

    def prepro_sentence_pair_single(self, input, max_length):

        if len(input) > max_length:
            input = input[len(input)-max_length:]
            assert len(input)==max_length

        n_mask = max_length - len(input)
        assert n_mask >= 0
        input_ids = input + [0 for _ in range(n_mask)]
        attention_mask = [1 for _ in input] + [0 for _ in range(n_mask)]

        return input_ids, attention_mask

    def tensorize(self, _train_data, _test_data, add_newlines=True):

        train_data, test_data = [], []
        if self.use_demonstrations:
            for dp in _train_data:
                assert type(dp)==dict, ("Each example should be a dictionary", dp)
                assert "input" in dp and "output" in dp, ("Training example should contain input and output", dp)
                if type(dp["output"])==str:
                    dp["output"] = [dp["output"]]
                train_data.append(dp.copy())

        for dp in _test_data:
            assert type(dp)==dict, ("Each example should be a dictionary", dp)
            assert "input" in dp and "output" in dp, ("Test example should contain input and output", dp)
            if type(dp["output"])==str:
                dp["output"] = [dp["output"]]
            test_data.append(dp.copy())

        input_ids = []
        attention_mask = []
        metadata = []

        if self.use_demonstrations:
            assert len(train_data)==self.k
            demonstrations = []
            for i, dp in enumerate(train_data):
                input_, output_ = self._prepro_each_datapoint(dp, is_first=i==0, for_demonstrations=True, add_newlines=add_newlines)
                demonstrations += input_ + output_

        for dp_idx, dp in enumerate(test_data):
            input_, output_ = self._prepro_each_datapoint(dp, is_first=not self.use_demonstrations, add_newlines=add_newlines)

            indices = [[i] for i in range(len(input_ids), len(input_ids)+len(input_))]

            metadata.append({"indices": indices})

            if self.use_demonstrations:
                input_ = demonstrations + input_

            input_ids_, attention_mask_ = self.prepro_sentence_pair_single(input_, self.max_length)
            input_ids.append(input_ids_)
            attention_mask.append(attention_mask_)

        self.tensorized_inputs = dict(input_ids=torch.LongTensor(input_ids), attention_mask=torch.LongTensor(attention_mask))
        self.metadata = metadata
