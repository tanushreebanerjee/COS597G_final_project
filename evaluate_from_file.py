import os
import json
import argparse
import numpy as np
from utils.our_data import evaluate

def main(prediction_path, groundtruth_path):

    predictions = []
    groundtruths = []

    with open(prediction_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            predictions.append(line)

    with open(groundtruth_path, "r") as f:
        for line in f:
            dp = json.loads(line)
            groundtruths.append(dp["output"])


    print(predictions[:5])
    print(groundtruths[:5])
    accs, f1s = evaluate(predictions, groundtruths)
    print("Accuracy: %f\nF1: %f" % (np.mean(accs) * 100, np.mean(f1s) * 100))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", required=True, type=str)
    parser.add_argument("--groundtruth_path", required=True, type=str)

    args = parser.parse_args()

    main(args.prediction_path, args.groundtruth_path)