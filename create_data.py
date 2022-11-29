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
    assert args.dataset in ['nq', 'squad']

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


    if args.variant in ["random_one_vocab", "random_length_vocab", "gibberish"]:
        from english_words import english_words_set
        english_words_set = sorted(english_words_set)

    new_data, orig_data = [], []
    with open(orig_path / f"{args.type}_orig.jsonl", "r") as fin: # either train_orig.jsonl or test_orig.jsonl
        for k, example in enumerate(fin):
            example = json.loads(example)
            orig_data.append(example)


    ### randomly choose k data 
    k = int(args.k)
    n = len(orig_data)
    indices = np.random.choice(n, k)

    if args.variant in ["random_one_vocab", "random_length_vocab"]:
        ### randomly choose one or length english words in the vocabulary as answers 

        for i in indices:
            dp = orig_data[i]

            # choose random words
            dp_ans_new = []
            for ans in dp["answer"]:
                dp_ans_orig_len = len(ans.split())
                if args.variant == "random_one_vocab": 
                    dp_ans_orig_len = 1
                rand_words = list(np.random.choice(english_words_set, size = dp_ans_orig_len, replace = False))
                dp_ans_new.append(" ".join(rand_words))

            new_data.append({
                "input": dp["question"],
                "output": dp_ans_new
            })

    elif args.variant == "permute":
        curr_batch = [orig_data[i] for i in indices]
        curr_batch_ans = [dp["answer"] for dp in curr_batch]
        np.random.shuffle(curr_batch_ans)

        for idx, dp in enumerate(curr_batch):
            new_data.append({
                "input": dp["question"], 
                "output": curr_batch_ans[idx]
            })

    elif args.variant in ["random_one", "random_length"]:
        ### Permute sentences given a context

        assert args.dataset == "squad"

        stop_words = set(stopwords.words("english"))

        for i in indices:
            dp = orig_data[i]
            dp_ans_new = []
            for ans in dp["answer"]:
                dp_ans_orig_len = len(ans.split())
                if args.variant == "random_one": 
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

            new_data.append({
                "input": dp["context"] + " Question: " +  dp["question"], 
                "output": dp_ans_new    
            })

    elif args.variant == "gibberish":
        ### Randomly insert one gibberish sentence

        assert args.dataset == "squad"

        for i in indices:
            dp = orig_data[i]

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

            new_data.append({
                "input": dp["context"] + " Question: " + dp["question"], 
                "output": dp["answer"]
            })
    
    if not os.path.isdir(orig_path / args.variant):
        os.makedirs(orig_path / args.variant)

    with open(orig_path / args.variant / f"{args.dataset}_{args.k}_{args.seed}_{args.type}.jsonl", "w") as f:
        for dp in new_data:
            f.write(json.dumps(dp) + "\n")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data")
    
    parser.add_argument("--dataset", type=str, default=None, required=True, help="nq or squad")
    parser.add_argument("--type", type=str, default="train", help="The type of data: train or test")
    parser.add_argument("--k", type=int, default=16, help="Number of demonstrations")
    parser.add_argument("--seed", type=str, default="42")
    parser.add_argument("--variant", type=str, default="random", required=True)
    parser.add_argument("--repeat_times", type=str, default="None")
    
    
    args = parser.parse_args()
    main(args)