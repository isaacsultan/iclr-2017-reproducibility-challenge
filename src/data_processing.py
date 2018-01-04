import load
import tokenizer
import pickle
import numpy as np
from collections import Counter
import pandas
import os

tags = ["eou", "eot"]
word_counts_path = "dumps/word_counts.pkl"
word_indices_parth = "dumps/word_indices.pkl"

_unk = "<UNK>"
_pad = "<PAD>"

def construct_indices_from_count():
    """Convert the dictionary of word counts into a dictionary of word indices"""
    with open(word_counts_path, "rb") as f:
        counts = pickle.load(f)

    vocab = list(counts.keys())
    # Account for padding and unknown words
    vocab = [_pad, _unk] + vocab
    word_indices = dict(zip(vocab, range(len(vocab))))

    with open(word_indices_parth, "wb") as f:
        pickle.dump(word_indices, f)


def reconstruct_tags(sentences):
    """Tags in the form __tag__ are being tokenize into 3 tokens. 
    We don't want that to happen, so we put them back together"""
    new_sents = []
    for sentence in sentences:
        temp_sent = np.array(sentence)
        to_remove = []
        for tag in tags:
            indices = np.argwhere(temp_sent == tag).flatten()

            for i in indices:
                if temp_sent[i-1] == "__" and temp_sent[i+1] == "__":
                    to_remove.extend([i-1, i+1])
                    temp_sent[i] = "__" + tag + "__"

        new_sents.append(np.delete(temp_sent, to_remove).tolist())

    return new_sents

def merge_back_test_array(context, true, distractors):
    res = []
    for i in range(len(context)):
        row = []
        row.append(context[i])
        row.append(true[i])
        for k in range(len(distractors)):
            row.append(distractors[k][i])
        
        res.append(row)

    return res

def merge_back_train_array(context, hypothesis, value):
    # Value is a numpy array, so use item() to get the value
    res = []
    for i in range(len(context)):
        row = []
        row.append(context[i])
        row.append(hypothesis[i])
        row.append(value[i])
        res.append(row)
    return res


def split_training_dataset(file, nb_splits, output_format_file):
    # Output format file is expected to be in the form "filename_{}.csv", where the brackets will be replaced by the split number
    train = load.load_csv(file)

    subtrains = np.split(train, nb_splits, 0)

    for i in range(len(subtrains)):
        df = pandas.DataFrame(subtrains[i])
        df.to_csv(output_format_file.format(i+1), header=["Context", "Utterance" , "Label"], index=False, encoding="utf-8")


def tokenize_dataset(file_in, is_training, file_out):
    
    if os.path.isfile(file_out):
        return

    # Load dict from pickle
    if os.path.isfile(word_counts_path):
        with open(word_counts_path, "rb") as f:
            words = pickle.load(f)
    else:
        words = Counter()

    # Load all data, tokenize, fetch all unique words
    print("Loading file...")
    dataset = load.load_csv(file_in)
    print("Done.")

    print("Preprocess sentences...")
    col_range = range(2) if is_training else range(11)
    results = []

    for i in col_range:
        res = tokenizer.tokenize_all(dataset[:,i])
        res = reconstruct_tags(res)
        
        for sentence in res:
            words.update(sentence)

        results.append(res)
    print("Done.")

    print("Dumping dictionary of words and tokenized dataset...")
    # Dump word dictionary
    with open(word_counts_path, "wb") as f:
        pickle.dump(words, f)

    #Merge back in correct form
    if is_training:
        results = merge_back_train_array(results[0], results[1], dataset[:,2])
    else:
        results = merge_back_test_array(results[0], results[1], results[2:11])
    with open(file_out, "wb") as f:
        pickle.dump(results, f)




if __name__ == "__main__":

    for i in range(1, 11):
        tokenize_dataset("data/ubuntu_train_{}.csv".format(i), True, "dumps/train_{}.pkl".format(i))

    tokenize_dataset("data/ubuntu_valid.csv", False, "dumps/valid.pkl")
    tokenize_dataset("data/ubuntu_test.csv", False, "dumps/test.pkl")

    # Store word indices
    construct_indices_from_count()








