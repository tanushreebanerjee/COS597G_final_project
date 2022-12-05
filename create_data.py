import os
import argparse
import random
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def main(args):

    datasets = args.dataset.split(",")
    variants = args.variant.split(",")
    seeds = args.seed.split(",")

    for dataset in datasets:
        assert dataset in ['nq', 'squad']

    for variant in variants:
        assert variant in [
            "gold",
            "random_one", "random_length", # randomly select from current context
            "permute", # permute in current sample context
            "random_one_vocab", "random_length_vocab", # randomly select from vocabulary, might be ood
            ### change input demonstration <- squad dataset
            "repeat_one_sent", # S1, S2, S2, S2, S2, S3, S4
            "gibberish" # S1, gibberish, S2, S3, gibberish, S4
        ]

    for dataset in datasets:
        for variant in variants:
            for seed in seeds:
                    print("Creating data for dataset: %s, variant: %s, seed: %s" % (dataset, variant, seed))
                    if variant == "gold":
                        create_gold_data(dataset, args.k, int(seed))
                    else:
                        create_data(dataset, variant, args.k, int(seed), "train")

                    create_test_data(dataset, variant, args.k, int(seed))

def create_test_data(dataset, variant, k, seed):
    output_path = os.path.join("data", dataset, variant, "{}_{}_{}_{}.jsonl".format(dataset, k, seed, "test"))
    if os.path.exists(output_path):
        print("File %s already exists" % output_path)
        return

    orig_test_data = []
    input_path = os.path.join("data", dataset, "test_orig.jsonl")
    assert os.path.exists(input_path)

    with open(input_path, "r") as f:
        for line in f:
            dp = json.loads(line)
            orig_test_data.append(dp)

    
    new_test_data = []

    for dp in orig_test_data[:10000]:

        if not "answer" in dp:
            assert "answers" in dp
            dp["answer"] = dp["answers"]

        if dataset == "squad":
            new_test_data.append({
                "context": dp["context"],
                "input": dp["question"],
                "output": dp["answer"]
            })
        elif dataset == "nq":
            new_test_data.append({
                "input": dp["question"],
                "output": dp["answer"]
            })

    if not os.path.isdir(os.path.join("data", dataset, variant)):
        os.makedirs(os.path.join("data", dataset, variant))

    with open(output_path, "w") as f:
        for dp in new_test_data:
            f.write(json.dumps(dp) + "\n")
    

def create_gold_data(dataset, k, seed):

    train_data_path = os.path.join("data", dataset, "gold", "{}_{}_{}_{}.jsonl".format(dataset, k, seed, "train"))
    if os.path.exists(train_data_path):
        print("File %s already exists" % train_data_path)
        return

    orig_train_data_path = os.path.join("data", dataset, "train_orig.jsonl")
    assert os.path.exists(orig_train_data_path)

    orig_train_data = []

    with open(orig_train_data_path, "r") as f: # either train_orig.jsonl or test_orig.jsonl
        for line in f:
            dp = json.loads(line)
            orig_train_data.append(dp)
    print(orig_train_data[0])

    ### randomly choose k data
    k = int(k)
    n = len(orig_train_data)
    assert k <= n
    indices = np.random.choice(n, k)

    new_train_data = []

    for i in indices:
        dp = orig_train_data[i]

        if not "answer" in dp:
            assert "answers" in dp
            dp["answer"] = dp["answers"]


        if dataset == "squad":
            new_train_data.append({
                "context": dp["context"],
                "input": dp["question"],
                "output": dp["answer"]
            })
        elif dataset == "nq":
            new_train_data.append({
                "input": dp["question"],
                "output": dp["answer"]
            })

    if not os.path.isdir(os.path.join("data", dataset, "gold")):
        os.makedirs(os.path.join("data", dataset, "gold"))

    with open(train_data_path, "w") as f:
        for dp in new_train_data:
            f.write(json.dumps(dp) + "\n")


def create_data(dataset, variant, k, seed):
    np.random.seed(int(seed))

    data_path = os.path.join("data", dataset, variant, "{}_{}_{}_{}.jsonl".format(dataset, k, seed, "train"))
    if os.path.exists(data_path):
        print("File %s already exists" % data_path)
        return

    gold_data_path = os.path.join("data", dataset, "gold", "{}_{}_{}_{}.jsonl".format(dataset, k, seed, "train"))
    if not os.path.exists(gold_data_path):
        create_gold_data(dataset, k, seed)
    assert os.path.exists(gold_data_path)


    if variant in ["random_one_vocab", "random_length_vocab", "gibberish"]:
        from english_words import english_words_set
        english_words_set = sorted(english_words_set)

    new_data, orig_data = [], []
    with open(gold_data_path, "r") as f: # either train_orig.jsonl or test_orig.jsonl
        for line in f:
            dp = json.loads(line)
            orig_data.append(dp)

    assert len(orig_data) == k

    if variant in ["random_one_vocab", "random_length_vocab"]:
        for dp in orig_data:

            dp_ans_new = []
            for ans in dp["output"]:
                dp_ans_orig_len = len(ans.split())
                if variant == "random_one_vocab":
                    dp_ans_orig_len = 1
                rand_words = list(np.random.choice(english_words_set, size = dp_ans_orig_len, replace = False))
                dp_ans_new.append(" ".join(rand_words))

            dp["output"] = dp_ans_new
            new_data.append(dp)

    elif variant == "permute":
        answers = [dp["output"] for dp in orig_data]
        np.random.shuffle(answers)

        for idx, dp in enumerate(orig_data):

            dp["output"] = answers[idx]
            new_data.append(dp)

    elif variant in ["random_one", "random_length"]:
        ### Permute sentences given a context

        assert "context" in orig_data[0]

        stop_words = set(stopwords.words("english"))

        for dp in orig_data:
            dp_ans_new = []
            for ans in dp["output"]:
                dp_ans_orig_len = len(ans.split())
                if variant == "random_one":
                    dp_ans_orig_len = 1
                context = re.sub(r'[^\w\s]', '', dp["context"]) # remove all punctuations
                all_context_words = word_tokenize(context)
                filtered_context_words = [w for w in all_context_words if not w.lower() in stop_words]

                ### follow distribution of context; may require a change
                filtered_context_words_counter = Counter(filtered_context_words)
                filtered_context_words_distribution = {word: filtered_context_words_counter[word] / len(filtered_context_words)
                                                       for word in filtered_context_words_counter}

                rand_words = list(np.random.choice(list(filtered_context_words_distribution.keys()),
                                                   size = dp_ans_orig_len,
                                                   p = list(filtered_context_words_distribution.values())))

                dp_ans_new.append(" ".join(rand_words))

            dp["output"] = dp_ans_new
            new_data.append(dp)

    elif variant == "gibberish":
        ### Randomly insert one gibberish sentence

        assert "context" in orig_data[0]

        for dp in orig_data:
            sentences = dp["context"].split(".")
            idx = np.random.randint(len(sentences), size=1)[0]

            ### calculate mean of all sentences
            total_len = 0
            for sent in sentences:
                filtered_sent = re.sub(r'[^\w\s]', '', sent)
                total_len += len(word_tokenize(filtered_sent))
            mean_len = int(total_len / len(sentences))

            rand_words = list(np.random.choice(english_words_set, size = mean_len, replace = False))

            final_context = sentences[:idx] + [" ".join(rand_words) + ". "] + sentences[idx:]

            dp["context"] = ".".join(final_context)
            new_data.append(dp)


    if not os.path.isdir(os.path.join("data", dataset, variant)):
        os.makedirs(os.path.join("data", dataset, variant))

    with open(data_path, "w") as f:
        for dp in new_data:
            f.write(json.dumps(dp) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## list of datasets
    parser.add_argument("--dataset", type=str, default=None, required=True, help="nq or squad")
    parser.add_argument("--k", type=int, default=16, help="Number of demonstrations")
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--variant", type=str, default="random", required=True)
    parser.add_argument("--repeat_times", type=str, default="None")


    args = parser.parse_args()
    main(args)
