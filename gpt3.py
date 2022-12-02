import time
import sys
import numpy as np
import torch
import json
import openai
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import GPT2Tokenizer

class GPT3Model(object):

    def __init__(self, model_name, api_key, logger=None):
        self.model_name = model_name
        try:
            openai.api_key = api_key
        except Exception:
            pass
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.logger=logger


    def prepare_data(self, train_data, test_data, batch_size=10, dp_sep="\n", max_length=128):
        # format demonstrations
        demonstrations = ""
        for dp in train_data:
            if type(dp["output"]) == list:
                dp["output"] = dp["output"][0]

            if "context" in dp:
                demonstrations += "Context: " + dp["context"] + "\n" + "Question: " + dp["input"] + "\n" + "Answer: " + dp["output"][0] + "\n\n\n"
            else:
                demonstrations += "Question: " + dp["input"] + "\n" + "Answer: " + dp["output"][0] + "\n\n\n"

        # append demonstrations and separate options
        inputs = []
        outputs = []
        for dp in test_data:
            if "context" in dp:
                prompt = "Context: " + dp["context"] + "\n" + "Question: " + dp["input"] + "\n"
            else:
                prompt = "Question: " + dp["input"] + "\n"

            inputs += [demonstrations + prompt]

        # truncate inputs
        for i, input in enumerate(inputs):
            input_ids = self.tokenizer.encode(input)
            if (len(input_ids) > max_length):
                input_ids = input_ids[len(input_ids) - max_length:]
                assert len(input_ids) == max_length
            inputs[i] = self.tokenizer.decode(input_ids)

        if self.logger is not None:
            self.logger.info("Checking the first example...")
            self.logger.info(inputs[0] + "" + outputs[0])

        # construct a dataloader
        dataset = zip(inputs, outputs)
        input_chunks = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        dataloader = [input_chunks[i] for i in range(0, len(input_chunks))]

        return dataloader


    def do_predict(self, dataloader, max_len):
        predictions = []
        cache = []
        for inputs in dataloader:
            response = self.gpt3(inputs, max_len=max_len)

            cache.append((inputs, response))

            for choice in response["choices"]:
                predictions.append(choice["text"])

        return predictions, cache


    def gpt3(self, prompt, max_len=0, temp=0, num_log_probs=0, echo=True, n=1):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        while not received:
            try:
                response = openai.Completion.create(engine=self.model_name,
                                                    prompt=prompt,
                                                    max_tokens=max_len,
                                                    temperature=temp,
                                                    logprobs=num_log_probs,
                                                    echo=echo,
                                                    stop='\n',
                                                    n=n)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response