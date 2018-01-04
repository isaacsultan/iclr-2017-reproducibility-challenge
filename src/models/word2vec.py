import os
import pickle

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

"""
    Possibly useful resources:
    https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    https://rare-technologies.com/word2vec-tutorial/
    https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/
"""


def __train__(sentences, model=None):
    """Sentence input should be in format: 
    [['first', 'sentence'], ['second', 'sentence'], ..., ['last', 'sentence']]
    """

    # initialize model
    if model is None:
        if os.path.isfile(fname):
            # continue training with loaded model
            model = Word2Vec.load(fname)
            model.build_vocab(sentences, update=True)
        else:
            # New model
            model = Word2Vec(None, size=100, window=5, min_count=5, workers=4, iter=20)
            model.build_vocab(sentences)
    else:
        model.build_vocab(sentences, update=True)

    # Train
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    return model


def train(save_to="dumps/train_word2vec.model"):
    model = None
    # Load sentences by batch
    for i in range(1,11):

        print("Loading batch {}".format(i))
        with open("dumps/train_{}.pkl".format(i), "rb") as f:
            train_set = pickle.load(f)
        print("Done")

        print("Training word2vec on batch {}".format(i))
        df_train = pd.DataFrame(train_set)

        # TODO: if value is one, concatenate both context and hypothesis ?

        # Context
        model = __train__(df_train[0].values, model=model)
        # Hypothesis
        model = __train__(df_train[1].values, model=model)
        print("Done.")

    # Finally, save the model
    model.save(save_to)


if __name__ == "__main__":
    train()
    print("All training sets have been trained. Result in dumps/train_word2vec.model")
