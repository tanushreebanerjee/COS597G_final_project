import os
import argparse
import random
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re

def main(args):
    assert args.variant in [
        "gold", 
        "random_one", "random_length", # randomly select from current context
        "permute", # permute in current sample context
        "random_one_vocab", "random_length_vocab", # randomly select from vocabulary, might be ood
        ### change input demonstration <- squad dataset
        "repeat_one_sent", # S1, S2, S2, S2, S2, S3, S4 
        "gibberish" # S1, gibberish, S2, S3, gibberish, S4
    ]
    
    np.random.seed(int(args.seed))
    
    if args.variant == "gold":
        print ("No need to run `create_data.py` --- you can use the original data as it is.")
        return
    
    dir_path = Path(args.data_dir)
    orig_path = dir_path / args.dataset # nq or squad
    
    if args.variant in ["random_one_vocab", "random_length_vocab"]:
        from english_words import english_words_set
        english_words_set = sorted(english_words_set)
    
    
    train_data = []
    with open(orig_path / "train_orig.jsonl", "r") as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            train_data.append(example)
    
    if args.variant in ["random_one_vocab", "random_length_vocab"]:
        
        for i, dp in enumerate(train_data):
            dp_ans_new = []
            for ans in dp["answer"]:
                dp_ans_orig_len = len(ans.split())
                if args.variant == "random_one_vocab": 
                    dp_ans_orig_len = 1
                rand_words = list(np.random.choice(english_words_set, size = dp_ans_orig_len, replace = False))
                dp_ans_new.append(" ".join(rand_words))
            
            dp["answer"] = dp_ans_new
    
    elif args.variant == "permute":
        k = int(args.k)
        for i in range(0, len(train_data), k):
            curr_batch = train_data[i:i+k+1]
            curr_batch_ans = [dp["answer"] for dp in curr_batch] # this creates a new list
            np.random.shuffle(curr_batch_ans)
            
            for idx, dp in enumerate(curr_batch):
                dp["answer"] = curr_batch_ans[idx]
    
    elif args.variant in ["random_one", "random_length"]:
        ### download first
        # import nltk
        # nltk.download('stopwords')
        # nltk.download('punkt')
        
        ### "random_one" and "random_length" are only for squad dataset

        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        stop_words = set(stopwords.words('english'))   
        
        for i, dp in enumerate(train_data):
            dp_ans_new = []
            for ans in dp["answer"]:
                dp_ans_orig_len = len(ans.split())
                if args.variant == "random_one": 
                    dp_ans_orig_len = 1
                context = re.sub(r'[^\w\s]', '', dp["context"]) # remove all punctuations
                all_context_words = word_tokenize(context)
                filtered_context_words = [w for w in all_context_words if not w.lower() in stop_words]
                
                ### follow distribution of context
                filtered_context_words_counter = Counter(filtered_context_words)
                filtered_context_words_distribution = {word: filtered_context_words_counter[word] / len(filtered_context_words) 
                                                       for word in filtered_context_words_counter}
                
                rand_words = list(np.random.choice(list(filtered_context_words_distribution.keys()), 
                                                   size = dp_ans_orig_len, 
                                                   p = list(filtered_context_words_distribution.values())))
                
                dp_ans_new.append(" ".join(rand_words))
            dp["answer"] = dp_ans_new
    
    elif args.variant == "permute_context":
        ### Permute sentences given a context
        
        for i, dp in enumerate(train_data):
            context = re.sub(r'[^\w\s]', '', dp["context"])
            all_context_words = word_tokenize(context)
            filtered_context_words = [w for w in all_context_words if not w.lower() in stop_words]
            
            
    with open(orig_path / f"train_{args.variant}_{args.seed}_{args.k}.jsonl", "w") as f:
        for dp in train_data:
            f.write(json.dumps(dp))
            f.write("\n")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--variant", type=str, default="random", required=True)

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--corpus_path", type=str, default=None)

    args = parser.parse_args()

    main(args)