import os
import csv
import json
import string
import numpy as np
import torch

def load_data(split, k, datasets, seed=0):
    data = {}
    for dataset in datasets:
        curr_data = []

        ## CHANGE WITH BEIQI
        data_path = os.path.join("data", dataset, "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                curr_data.append(dp)
        data[dataset] = curr_data
    return data
