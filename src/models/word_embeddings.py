import pickle
import csv
import os

from collections import Counter
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np

word_indices_path = "dumps/word_indices.pkl"
enhanced_we_path = "dumps/enhanced_we.pkl"

def load_embedding_matrix(we: dict, embedding_dimension: int):
    """
    Given word embeddings, compute the related embedding matrix
    """
    with open(word_indices_path, "rb") as f:
        word_indices = pickle.load(f)

    # Defaults to 0 if not found
    embedding = np.zeros((len(word_indices), embedding_dimension))

    for key, val in we.items():
        if key in word_indices:
            embedding[word_indices[key]] = val

    return embedding, word_indices



def compute_enhanced(words: set, word2vec: Word2Vec, pretrained: pd.DataFrame):
    """
    Algorithm to compute the enhanced word representation
    :param dictionary: Counter object which holds all unique tokens
    :param word2vec: Word2Vec object representing the trained word2vec model
        from the Word2Vec object, keyedvector representation can be found in word2vec_obj.wv
    :param pretrainedGlove: Dataframe containing the pretrained glove word embeddings

    :return : A dictionary of words as key as word embeddings as value
    """
    enhanced = dict()
    vec_size = pretrained.shape[1] + word2vec.vector_size

    # Store indexing. Pretrained comes first
    u_begin = 0
    u_end = pretrained.shape[1]
    v_begin = u_end
    v_end = v_begin + word2vec.vector_size

    for word in words:
        res = np.zeros((vec_size))
        if word in pretrained.index:
            res[u_begin:u_end] = pretrained.loc[word]

        if word in word2vec:
            res[v_begin:v_end] = word2vec[word]

        if np.count_nonzero(res) > 0:
            enhanced[word] = res
    return enhanced


def load_and_compute_enhanced():
    print("Computing enhanced word embeddings")

    with open("dumps/word_counts.pkl", "rb") as f:
        word_counts = pickle.load(f)
    words = set(word_counts.keys())

    w2vmodel = Word2Vec.load("dumps/train_word2vec.model")
    pretrained = pd.read_csv('data/glove_filtered.txt', header=None, 
                index_col=0, delim_whitespace=True, quoting=csv.QUOTE_NONE)

    res = compute_enhanced(words, w2vmodel, pretrained)

    with open(enhanced_we_path, "wb") as f:
        pickle.dump(res, f)

    print("Done.")
    return res

if __name__ == "__main__":

    # Load or compute enhanced word embeddings
    if os.path.isfile(enhanced_we_path):
        with open(enhanced_we_path, "rb") as f:
            enhanced = pickle.load(f)
    else:
        enhanced = load_and_compute_enhanced()

    # Create fixed embedding matrix
    embed_matrix, word_indices = load_embedding_matrix(enhanced, 400)

    

