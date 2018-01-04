import pickle
import csv
import pandas as pd
import numpy as np

print("Read files...")
iterator = pd.read_csv('data/glove.42B.300d.txt', header=None, index_col=0, 
                        delim_whitespace=True, quoting=csv.QUOTE_NONE, dtype="str", chunksize=100000)

with open('dumps/word_counts.pkl', 'rb') as f:
    word_dict = pickle.load(f)
print("Done.")

df = pd.DataFrame()

words = set(word_dict.keys())

total = 0
in_glove = 0
total_ubuntu = len(words)

print("Iterating through chunks...")
done = 0
# Iterate chunk by chunk
for i in iterator:
    total += i.shape[0]
    unique_toks = set(i.index.values)
    in_glove += len(unique_toks.intersection(words))

    remain = unique_toks - words
    df = df.append(i.drop(remain, axis=0))
    done += 1
    print("Batch {} done".format(done))
print("Done.")

# Print compression percentage
filtered = df.shape[0]
print("Kept {0:.4f}% of the rows".format((filtered/total) * 100))
print("{0:.4f}% of tokens were in glove".format(in_glove/total_ubuntu))

df.to_csv("data/glove_filtered.txt", sep=" ", header=False, index=True, quoting=csv.QUOTE_NONE)