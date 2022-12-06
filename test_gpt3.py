import os
import time
import sys
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np
import openai


from tqdm import tqdm
from collections import Counter, defaultdict

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import GPT2Tokenizer, AutoTokenizer

from gpt3 import GPT3Model

from utils.our_data import load_data, evaluate

def main(logger, args):

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    add_newlines = False
    checkpoint = None
        	
    gpt3_model = GPT3Model(args.gpt3, args.api, logger)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data

    max_length_per_example = 256
    max_length = 256
    if args.use_demonstrations:
        orig_max_length = max_length
        max_length = min(max_length * args.k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    accs = []
    f1s = []
    errors = []
    seeds = args.seed.split(",")
    datasets = args.dataset.split(",")

    for seed in seeds:

        train_data = load_data("train", args.k, datasets, args.variant, seed=seed)
        test_data = load_data("test", args.k, datasets, args.variant, seed=seed)

        for dataset in datasets:
            curr_train_data = train_data[dataset]
            curr_test_data = test_data[dataset]

            ## NUMBER OF DEMONSTRATIONS SHOULD MATCH THE ARGUMENT PROVIDED
            assert not args.use_demonstrations or len(curr_train_data)==args.k

            logger.info("gpt3 - %s on %s (%d train, %d dev)" % (args.gpt3, args.dataset, len(curr_train_data), len(curr_test_data)))

            result = run(logger, dataset, gpt3_model, curr_train_data, curr_test_data, seed, max_length)
            #print("result", result)
            result = list(result)
            if result is None:
                errors.append("%s/%s" % (dataset, seed))
            else:
                accs.append(result[0])
                f1s.append(result[1])
                #results.append(result)

    print("Macro-F1 of %s over %d target tasks: %.1f" % (args.dataset, len(f1s) // len(seeds), 100 * np.mean(f1s)))
    print("Accuracy of %s over %d target tasks: %.1f" % (args.dataset, len(accs) // len(seeds), 100 * np.mean(accs)))

    logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.dataset, len(f1s) // len(seeds), 100 * np.mean(f1s)))
    logger.info("Accuracy of %s over %d target tasks: %.1f" % (args.dataset, len(accs) // len(seeds), 100 * np.mean(accs)))

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")


def run(logger, dataset, gpt3_model, train_data, test_data, seed, max_length):

    cache_path = os.path.join(args.out_dir,
                              "{}-{}{}{}.pkl".format(
                                  dataset,
                                  "test",
                                  "-k={}".format(args.k) if args.use_demonstrations else "",
                                  "-s={}".format(seed)))


    logger.info(cache_path)
    gpt3_cache_path = cache_path.replace(".pkl", ".json")
    prediction_path = cache_path.replace(".pkl", ".txt")

    groundtruths = [dp["output"] for dp in test_data]
    max_gt_length = 0
    for groundtruth in groundtruths:
        for gt in groundtruth:
            max_gt_length = max(max_gt_length, len(gt))

    MAX_GENERATION_LENGTH = min(max_length + max_gt_length, 1024)

    dataloader = gpt3_model.prepare_data(train_data if args.use_demonstrations else [], test_data, batch_size=args.test_batch_size, max_length=max_length)
    predictions, cache = gpt3_model.do_predict(dataloader, MAX_GENERATION_LENGTH)

    with open(gpt3_cache_path, "w") as f:	
        json.dump(cache, f)

    for i, prediction in enumerate(predictions):
        lines = [line for line in prediction.split("\n") if line]
        if len(lines) == 0:
            lines = [""]
            
        has_ans = False
        ans_line_no = 0
        for j, line in enumerate(lines):
            if "Answer:" in line:
                ans_line_no = j
                has_ans = True
                break

        if not has_ans:
            predictions[i] = lines[0]
        else:
            line = lines[ans_line_no]
            index = line.index("Answer:")
            predictions[i] = line[index + 7:]

    with open(gpt3_cache_path, "w") as f:	
        json.dump(cache, f)

    accs, f1s = evaluate(predictions, groundtruths)
    logger.info("Accuracy=%s" % np.mean(accs))
    logger.info("F1=%s" % np.mean(f1s))

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return np.mean(accs), np.mean(f1s)

if __name__=='__main__':

    parser = argparse.ArgumentParser()


    ## WHETHER WE USE DEMONSTRATIONS OR NOT
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    ## SPECIFY PATH TO LOG FILE
    parser.add_argument("--log_file", default="logs", type=str)

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
    parser.add_argument("--gpt3", type=str, default="curie", choices=["ada", "babbage", "curie", "davinci"])

    parser.add_argument("--variant", type=str, default="gold", required=True)
    parser.add_argument("--api", type=str, default="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")

    args = parser.parse_args()

    logging.basicConfig(filename=f"{args.log_file}/{args.dataset}-{args.k}-{args.seed}-{args.variant}-logging.log", 
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)