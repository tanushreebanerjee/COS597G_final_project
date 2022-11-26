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
from transformers import GPT2Tokenizer, AutoTokenizer

from icl.data import Data
from icl.model import Model

from utils.our_data import load_data

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
    metaicl_model = MetaICLModel(logger, args.out_dir)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    max_length_per_example = 256
    max_length = 256
    if args.use_demonstrations:
        orig_max_length = max_length
        max_length = min(max_length * args.k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (args.test_batch_size, max_length, max_length_per_example))

    ## TODO: NEED TO CHANGE THIS LINE OF CODE
    metaicl_data = MetaICLData(logger, tokenizer, args.method, args.use_demonstrations, args.k, max_length, max_length_per_example)

    results = []
    errors = []
    seeds = args.seed.split(",")
    datasets = args.dataset.split(",")

    for seed in seeds:

        ### TODO: NEED TO CHANGE LOAD_DATA TO NOT ACCEPT "TASK" and "config_split"
        train_data = load_data("train", args.k, datasets, seed=seed)
        dev_data = load_data("test", args.k, datasets, seed=seed)

        if args.use_random_english_words:
            from english_words import english_words_set
            english_words_set = sorted(english_words_set)
            np.random.seed(int(seed))

        for dataset in datasets:
            curr_train_data = train_data[dataset]
            curr_dev_data = dev_data[dataset]

            ## NUMBER OF DEMONSTRATIONS SHOULD MATCH THE ARGUMENT PROVIDED
            assert not args.use_demonstrations or len(curr_train_data)==args.k

            logger.info("%s - %s on %s (%d train, %d dev)" % (args.gpt2, args.method, args.dataset, len(curr_train_data), len(curr_dev_data)))

            config_file = "config/tasks/{}.json".format(dataset)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)

            ## WHAT SHOULD I DO ABOUT THIS
            is_classification = config["task_type"]=="classification"
            if is_classification:
                options = curr_dev_data[0]["options"]
                assert np.all([d["options"]==options for d in curr_dev_data])

            ## SEEMS LIKE USELESS CODE
            if args.use_random_english_words:
                # create a mapping
                options = curr_dev_data[0]["options"]
                mapping = {option: np.random.choice(english_words_set) for option in options}
                new_options = list(mapping.values())
                for dp_idx, dp in enumerate(curr_train_data):
                    assert dp["output"] in options, (dp, options)
                    curr_train_data[dp_idx]["output"] = mapping[dp["output"]]
                    curr_train_data[dp_idx]["options"] = new_options
                for dp_idx, dp in enumerate(curr_dev_data):
                    assert dp["output"] in options, (dp, options)
                    curr_dev_data[dp_idx]["output"] = mapping[dp["output"]]
                    curr_dev_data[dp_idx]["options"] = new_options

            ## WILL NEED TO CHANGE
            result = run(logger, dataset, metaicl_data, metaicl_model, curr_train_data, curr_dev_data, seed, checkpoint, is_classification, add_newlines)

            if result is None:
                errors.append("%s/%s" % (test_task, seed))
            else:
                results.append(result)


    #logger.info("Macro-F1 of %s over %d target tasks: %.1f" % (args.task, len(results) // len(seeds), 100*np.mean(results)))

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")

def run(logger, dataset, metaicl_data, metaicl_model, train_data, dev_data, seed, checkpoint, is_classification, add_newlines):

    cache_path = os.path.join(args.out_dir,
                              "{}-{}-{}{}{}{}{}.pkl".format(
                                  task,
                                  "test",
                                  metaicl_data.method,
                                  "-k={}".format(args.k) if args.use_demonstrations else "",
                                  "-s={}".format(seed) if args.use_demonstrations or args.use_random_english_words else "",
                                  "" if add_newlines else "-no-newlines",
                                  "-randomEnglish" if args.use_random_english_words else ""))

    metaicl_data.tensorize(train_data, dev_data, add_newlines=add_newlines)
    metaicl_data.print_tensorized_example()
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")

    if os.path.exists(prediction_path):
        return 0

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            losses = pkl.load(f)
    else:
        if metaicl_model.is_none():
            metaicl_model.load(checkpoint, gpt2=args.gpt2)
            metaicl_model.cuda()
            metaicl_model.eval()

        ## NEED TO CHANGE
        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
        with open(cache_path, "wb") as f:
            pkl.dump(losses, f)

    assert len(losses)==len(metaicl_data)

    ## NEED TO CHANGE EVERYTHING
    predictions = metaicl_model.do_predict(metaicl_data, losses=losses)
    groundtruths = [dp["output"] for dp in dev_data]
    perf = metaicl_data.evaluate(predictions, groundtruths, is_classification)
    logger.info("Accuracy=%s" % perf)

    with open(prediction_path, "w") as f:
        for prediction in predictions:
            f.write(prediction)
            f.write("\n")

    return perf

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    ## MIGHT DELETE LATER
    #parser.add_argument("--do_zeroshot", default=False, action="store_true")
    ## WHETHER WE USE DEMONSTRATIONS OR NOT
    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    ## SPECIFY PATH TO LOG FILE
    parser.add_argument("--log_file", default=None, type=str)

    ## LIST OF DATASETS (e.g., QASC, COMMONSENSE_QA)
    parser.add_argument("--dataset", type=str, default=None)
    ## NUMBER OF DEMONSTRATIONS
    parser.add_argument("--k", type=int, default=16)
    ## RANDOM SEED
    parser.add_argument("--seed", type=str, default="100")
    ## SUGGESTED VALUES
    ## 64 / 16 for GPT-2 with no demonstrations / few-shot
    ## 16 / 4  for GPT-J with no demonstratiosn / few-shot
    parser.add_argument("--test_batch_size", type=int, default=64)
    ## STORED MODEL CHECKPOINT (NEEDED IF WE NEED TO RUN GPT-J)
    parser.add_argument("--checkpoint", type=str, default=None)

    ## MIGHT DELETE LATER
    parser.add_argument("--use_random_english_words", default=False, action="store_true")
    ## PATH TO OUTPUT
    parser.add_argument("--out_dir", type=str, required=True)
    ## WHAT IS THIS?
    #parser.add_argument("--split", type=str, default="test")
    #parser.add_argument("--is_null", default=False, action="store_true")
    ## SPECIFY THE MODEL TO RUN
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
