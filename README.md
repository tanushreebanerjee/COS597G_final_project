# COS597G_final_project
COS597G final project Fall 2022

**Final presentation: Dec 5**

File structure:

```
project
│   README.md
│   preprocess_squad.py // convert SQuAD json to jsonl    
│   create_data.py // prepare training data
```

---
## Dataset

Download the NQ dataset using: 

```
# Get NQ dataset
wget -c https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl -P data

wget -c https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.train.jsonl -P data
```

You can directly download SQuADv1.1 from [here](https://rajpurkar.github.io/SQuAD-explorer/). 

### Generate training data


Run the following command to generate corresponding training data: 
```
python create_data.py \
    --dataset {nq|squad} \
    --variant {gold|random_one|random_length|permute...} \ 
    --k {4|8|16|32} \
    --seed {42} \
    --repeat_times $repeat_times \
    --data_dir $data_dir
```

- `--dataset`: name of the dataset, either `nq` or `squad` for now, required to fill in
- `--variant`: name of different methods for generating training data
    - `gold`: original training data (`data/{dataset}/train_orig.jsonl`)
    - `random_one`: randomly select one word from current context as the answer (only used in SQuAD for now, because questions in NQ are too short and it doesn't make sense to select words from those short questions)
    - `random_length`: randonly select `length = len(original answer)` words from current context as the answer (only used in SQuAD for now)
    - `permute`: permute gold answers in `k` in-context samples
    - `random_one_vocab`: randomly select one word from english vocabulary as the answer
    - `random_length_vocab`: randomly select `length = len(original answer)` words from english vocabulary as the answer
    - `repeat_one_sent`: randomly select one sentence from the context and repeat that sentence for `repeat_times`. e.g. S1, S2, S3, S4 -> S1, S2, S2, S2, S2, S3, S4 (change the context)
    - `gibberish`: first randomly pick an index, and then insert a gibberish sentence with `length = average sentence length in the context`. e.g. S1, S2, S3 -> S1, gibberish, S2, S3 (change the context)
- `--k`: the number of in-context samples
- `--seed`: random seed, default is 42
- `--repeat_times`: used in `repeat_one_sent`, specifies how many times a selected sentence is repeated
- `--data_dir`: data directory

A sample usage: 

```
python create_data.py --dataset squad --variant repeat_one_sent --repeat_times 3
```
