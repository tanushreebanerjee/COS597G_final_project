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
from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, AutoModelForCausalLM
from our_data import GPT2Data

from utils.our_data import load_data, evaluate

from transformers import pipeline, set_seed

def main(logger, args):

    if args.gpt2.startswith("gpt2"):
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        #tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2)
        add_newlines = False
    else:
        # args.gpt2=="gpt-j-6B":
        # we are using the HF veresion where GPT-J-6B checkpoint is not officially registered
        # so need to download the model checkpoint and specify checkpoint
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        #tokenizer = AutoTokenizer.from_pretrained("gpt2")
        add_newlines = True
        assert args.checkpoint is not None and os.path.exists(args.checkpoint)
        args.gpt2 = args.checkpoint

    checkpoint = None

    ## TODO: NEED TO CHANGE THIS LINE OF CODE
    #metaicl_model = MetaICLModel(logger, args.out_dir)
    #gpt2_model = GPT2Model.from_pretrained('gpt2')
    gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    max_length_per_example = 128
    max_length = 128
    if args.use_demonstrations:
        orig_max_length = max_length
        max_length = min(max_length * args.k, 128 * 7)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (args.test_batch_size, max_length, max_length_per_example))

    ## TODO: NEED TO CHANGE THIS LINE OF CODE
    gpt2_data = GPT2Data(logger, tokenizer, args.use_demonstrations, args.k, max_length, max_length_per_example)
    accs = []
    f1s = []
    #results = []
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

            logger.info("%s on %s (%d train, %d dev)" % (args.gpt2, args.dataset, len(curr_train_data), len(curr_test_data)))

            result = run(logger, dataset, gpt2_data, gpt2_model, curr_train_data, curr_test_data, seed, checkpoint, add_newlines)
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

    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")

def run(logger, dataset, gpt2_data, gpt2_model, train_data, test_data, seed, checkpoint, add_newlines):

    cache_path = os.path.join(args.out_dir,
                              "{}-{}{}{}{}.pkl".format(
                                  dataset,
                                  "test",
                                  "-k={}".format(args.k) if args.use_demonstrations else "",
                                  "-s={}".format(seed) if args.use_demonstrations else "",
                                  "" if add_newlines else "-no-newlines"))

    gpt2_data.tensorize(train_data, test_data, add_newlines=add_newlines)
    print(gpt2_data.print_tensorized_example(return_string=True))
    logger.info(cache_path)
    prediction_path = cache_path.replace(".pkl", ".txt")

    # UNCOMMENT LATER!!
    # if os.path.exists(prediction_path):
    #     return 0

    # if os.path.exists(cache_path):
    #     with open(cache_path, "rb") as f:
    #         losses = pkl.load(f)
    # else:
    #     dataloader = gpt2_data.get_dataloader(args.test_batch_size, is_training=False)
    #     for batch in dataloader:
    #         input_ids = batch[0]
    #         attention_mask = batch[1]
    #         print(gpt2_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)

    #return 1

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

    groundtruths = [dp["output"] for dp in test_data]
    max_gt_length = 0
    for groundtruth in groundtruths:
        for gt in groundtruth:
            max_gt_length = max(max_gt_length, len(gt))

    MAX_GENERATION_LENGTH = min(gpt2_data.max_length + max_gt_length, 1024)

    gpt2_tokenizer = gpt2_data.tokenizer

    predictions = []
    dataloader = gpt2_data.get_dataloader(args.test_batch_size, is_training=False)
    for batch in dataloader:
        input_ids = batch[0]
        attention_mask = batch[1]
        generation_output = gpt2_model.generate(input_ids, attention_mask=attention_mask, do_sample=False, max_length=MAX_GENERATION_LENGTH, return_dict_in_generate=True)

        generated_sequences = generation_output.sequences
        idx = len(input_ids[0]) - len(generated_sequences[0])
        #print("idx", idx)
        generated_sequences = [generated_sequences[0][idx:]]

        decoded_outputs = gpt2_tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        for decoded_output in decoded_outputs:
            lines = [line for line in decoded_output.split("\n") if line]
            has_ans = False
            ans_line_no = 0
            for i, line in enumerate(lines):
                if "Answer: " in line:
                    ans_line_no = i
                    has_ans = True
                    break

            if not has_ans:
                predictions.append(lines[0])
            else:
                line = lines[ans_line_no]
                index = line.index("Answer: ")
                predictions.append(line[8:])



    # generate up to 30 tokens
    #print("len(input_ids[0])", len(input_ids[0]))

    # print("generated_sequences", generated_sequences)
    # print("len(generated_sequences)", len(generated_sequences), MAX_GENERATION_LENGTH)





    #print(predictions) #decoded_outputs

    # tanushree edits end Fri 25 Nov

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

    parser.add_argument("--variant", type=str, default="random", required=True)

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
