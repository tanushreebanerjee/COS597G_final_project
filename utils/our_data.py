import os
import csv
import json
import string
import numpy as np
import torch
from collections import Counter

def load_data(split, k, datasets, variant, seed=0):
    data = {}
    for dataset in datasets:
        curr_data = []

        ## CHANGE WITH BEIQI
        data_path = os.path.join("data", dataset, variant, "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                curr_data.append(dp)
        data[dataset] = curr_data
    return data

## https://kierszbaumsamuel.medium.com/f1-score-in-nlp-span-based-qa-task-5b115a5e7d41
## https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#Metrics-for-QA
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, groundtruth):
    return int(prediction in groundtruth)

def compute_f1(prediction, groundtruth):
    precisions = []
    recalls = []
    f1 = []

    prediction_words = prediction.split(" ")

    for gt in groundtruth:
        gt_words = gt.split(" ")
        common = Counter(prediction_words) & Counter(gt_words)
        print(common)
        num_same = sum(common.values())
        if len(prediction) == 0 or len(gt) == 0:
            f1.append(int(prediction == gt))
            continue

        if num_same == 0:
            f1.append(0)
            continue

        print("prediction: %s\n" % (prediction))
        print("groundtruth: %s\n" % (gt))
        print("num_same: %d\n" % (num_same))
        precision = 1.0 * num_same / len(prediction_words)
        recall = 1.0 * num_same / len(gt_words)
        print("precision: %f\n" % (precision))
        print("recall: %f\n" % (recall))

        f1.append(2 * precision * recall / (precision + recall))

    return np.max(f1)


def evaluate(predictions, groundtruths):
    accs = []
    f1s = []
    for prediction, groundtruth in zip(predictions, groundtruths):
        prediction = normalize_text(prediction)
        groundtruth = [normalize_text(gt) for gt in groundtruth] if type(groundtruth)==list else [normalize_text(groundtruth)]

        accs.append(compute_exact_match(prediction, groundtruth))


        f1s.append(compute_f1(prediction, groundtruth))

    return accs, f1s
