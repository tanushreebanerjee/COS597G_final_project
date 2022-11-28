import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np

from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer, GPT2Model

from our_data import GPT2Data

from utils.our_data import load_data

from transformers import pipeline, set_seed

def main(logger, args):

    if args.gpt2.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
        add_newlines = False
    else:
        # args.gpt2=="gpt-j-6B":
        # we are using the HF veresion where GPT-J-6B checkpoint is not officially registered
        # so need to download the model checkpoint and specify checkpoint
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        add_newlines = True
        assert args.checkpoint is not None and os.path.exists(args.checkpoint)
        args.gpt2 = args.checkpoint

    checkpoint = None

    ## TODO: NEED TO CHANGE THIS LINE OF CODE
    #metaicl_model = MetaICLModel(logger, args.out_dir)
    gpt2_model = GPT2Model.from_pretrained('gpt2')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    max_length_per_example = 256
    max_length = 256
    if args.use_demonstrations:
        orig_max_length = max_length
        max_length = min(max_length * args.k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (args.test_batch_size, max_length, max_length_per_example))

    ## TODO: NEED TO CHANGE THIS LINE OF CODE
    gpt2_data = GPT2Data(logger, tokenizer, args.method, args.use_demonstrations, args.k, max_length, max_length_per_example)

    results = []
    errors = []
    seeds = args.seed.split(",")
    datasets = args.dataset.split(",")

    for seed in seeds:

        train_data = load_data("train", args.k, datasets, seed=seed)
        dev_data = load_data("test", args.k, datasets, seed=seed)

        for dataset in datasets:
            curr_train_data = train_data[dataset]
            curr_dev_data = dev_data[dataset]

            ## NUMBER OF DEMONSTRATIONS SHOULD MATCH THE ARGUMENT PROVIDED
            assert not args.use_demonstrations or len(curr_train_data)==args.k

            logger.info("%s - %s on %s (%d train, %d dev)" % (args.gpt2, args.method, args.dataset, len(curr_train_data), len(curr_dev_data)))

            result = run(logger, dataset, gpt2_data, gpt2_model, curr_train_data, curr_test_data, seed, checkpoint, add_newlines)

            if result is None:
                errors.append("%s/%s" % (test_task, seed))
            else:
                results.append(result)


    print("Macro-F1 of %s over %d target tasks: %.1f" % (args.dataset, len(results) // len(seeds), 100 * np.mean(results)))
    logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.dataset, len(results) // len(seeds), 100 * np.mean(results)))

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")

def run(logger, dataset, gpt2_data, gpt2_model, train_data, test_data, seed, checkpoint, add_newlines):

    cache_path = os.path.join(args.out_dir,
                              "{}-{}-{}{}{}{}.pkl".format(
                                  dataset,
                                  "test",
                                  data.method,
                                  "-k={}".format(args.k) if args.use_demonstrations else "",
                                  "-s={}".format(seed) if args.use_demonstrations else "",
                                  "" if add_newlines else "-no-newlines"))

    gpt2_data.tensorize(train_data, dev_data, add_newlines=add_newlines)
    print(gpt2_data.print_tensorized_example(return_string=True))
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")

    if os.path.exists(prediction_path):
        return 0

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            losses = pkl.load(f)
    else:
        dataloader = gpt2_data.get_dataloader(args.test_batch_size, is_training=False)
        for batch in dataloader:
            input_ids = batch[0]
            attention_mask = batch[1]
            print(gpt2_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)

    return 1

    #tanushree edits start Fri 25 Nov
    #else:
        #dont think we need this?
        # if gpt2_model.is_none():
        #     gpt2_model.load(checkpoint, gpt2=args.gpt2)
        #     gpt2_model.cuda()
        #     gpt2_model.eval()

        ## NEED TO CHANGE
    #     losses = gpt2_model.do_inference(gpt2_data, args.test_batch_size)
    #     with open(cache_path, "wb") as f:
    #         pkl.dump(losses, f)

    # assert len(losses)==len(gpt2_data)

    ## NEED TO CHANGE EVERYTHING
    #predictions = gpt2_model.do_predict(gpt2_data, losses=losses)

    predictions = gpt2_model.predict(**gpt2_data)
    # tanushree edits end Fri 25 Nov

    groundtruths = [dp["output"] for dp in dev_data]
    perf = gpt2_data.evaluate(predictions, groundtruths, is_classification)
    logger.info("Accuracy=%s" % perf)

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return perf

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    ## WHETHER WE USE DEMONSTRATIONS OR NOT
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    ## SPECIFY PATH TO LOG FILE
    parser.add_argument("--log_file", default=None, type=str)

    ## LIST OF DATASETS (e.g., QASC, COMMONSENSE_QA)
    parser.add_argument("--dataset", type=str, default=None)
    ## NUMBER OF DEMONSTRATIONS
    parser.add_argument("--k", type=int, default=16)
    ## RANDOM SEED
    parser.add_argument("--seed", type=str, default="42")
    ## SUGGESTED VALUES
    ## 64 / 16 for GPT-2 with no demonstrations / few-shot
    ## 16 / 4  for GPT-J with no demonstratiosn / few-shot
    parser.add_argument("--test_batch_size", type=int, default=1)
    ## STORED MODEL CHECKPOINT (NEEDED IF WE NEED TO RUN GPT-J)
    parser.add_argument("--checkpoint", type=str, default=None)

    ## PATH TO OUTPUT
    parser.add_argument("--out_dir", type=str, required=True)
    ## SPECIFY THE MODEL TO RUN
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    parser.add_argument("--method", type=str, default="direct")

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(filename="logger.log",
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
