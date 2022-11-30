import sys
import json
from pathlib import Path
import numpy as np

dir_path = Path("data")
squad_orig_path = dir_path / 'squad'

with open(squad_orig_path / 'train-v1.1.json') as fin:
    train_orig = json.load(fin)["data"]

with open(squad_orig_path / 'train_orig.jsonl', 'w') as f:
    for idx, dp in enumerate(train_orig):
        
        for p in dp["paragraphs"]:
            for qa in p["qas"]:
                ans = qa["answers"]
                unique_ans = list(set([x["text"] for x in ans]))
                
                new_dp = {
                    "context": p["context"], 
                    "question": qa["question"],
                    "answer": unique_ans, 
                    "answers_orig": ans
                }
                
                f.write(json.dumps(new_dp))
                f.write("\n")
            