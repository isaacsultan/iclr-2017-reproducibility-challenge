# Load files into in-memory datasets
import pickle
import os
import numpy as np
import pandas as pd
import random
import csv

glove_pretrained_file = "./data/glove.42B.300d.txt"
glove_smaller_file = "./data/globe.6B.300d.txt"

def load_csv(file):
    """Load a csv file with commas as delimiters, and no headers
    return value as a numpy array"""
    data = pd.read_csv(file, encoding="utf-8", keep_default_na=False, header=0)
    return data.values

def breakdown(data):
    """Break down each row into context-utterance pairs.
    Each pair is labeled to indicate truth (1.0) vs distraction (0.0).
    Output is a numpy array with format: [context, utterance, label]"""
    output = []
    for row in data:
        context = row[0]
        ground_truth_utterance = row[1]
        output.append([list(context), ground_truth_utterance, 1])
        for i in range(2,11):
            output.append([list(context), row[i], 0.0])
    return output
        

def load_pretrained_glove(file, dump_file):
    """Load the pretrained filtered glove and return it as a dictionary"""

    # If the dump file exists, load this instead
    if os.path.isfile(dump_file):
        with open(dump_file, "rb") as f:
            return pickle.load(f)

    pretrained = pd.read_csv(file, header=None, 
            index_col=0, delim_whitespace=True, quoting=csv.QUOTE_NONE)

    pretrained_dict = dict()
    for i in pretrained.index:
        pretrained_dict[i] = pretrained.loc[i].values

    with open(dump_file, "wb") as f:
        pickle.dump(pretrained_dict, f)

    return pretrained_dict


if __name__ == "__main__":
    # Get validation and test pickle file, transform them into same format as training set
    to_process = [["dumps/valid.pkl", "dumps/valid_expanded.pkl"], ["dumps/test.pkl", "dumps/test_expanded.pkl"]]

    for t in to_process:
        with open(t[0], "rb") as f:
            valid = pickle.load(f)
    
        new_valid = breakdown(valid)

        with open(t[1], "wb") as f:
            pickle.dump(new_valid, f)