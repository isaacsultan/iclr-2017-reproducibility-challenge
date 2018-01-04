import pickle
import numpy as np
import pandas as pd

import load
import train_model as model
import util.evaluate as eval
import models.word_embeddings as we


def indices_gen(array_length):
    index = 0
    indices = []
    for i in range(array_length):
        indices.append(index)
        if (i+1) % 10 == 0:
            index += 1
    return indices

def join_rank_predictions(y_true, y_predict, indices):
    df = pd.DataFrame({
        'context_indices': indices,
        'response': y_true[:,1],
        'likelihood': y_predict,
        'label': y_true[:,2]
    })
    sorted_y = df.sort_values(
        by=['context_indices', 'likelihood'], ascending=[True, False])
    return sorted_y


def rank_predictions(X_matrix, predictions, labels, label_col=2):
    df = pd.DataFrame({'context': X_matrix[0], 'response': X_matrix[1], 'prediction': predictions})
    sorted_df = df.sort_values(by=['context', 'prediction'], ascending=[True, False])
    return sorted_df


def recall_at_k(predictions, K=1, ranks=10):
    """Recall = TP/(TP+FN)"""
    tp, fn = 0.0, 0.0
    itervals = predictions.values
    label_idx = predictions.columns.get_loc("label")
    N = itervals.shape[0]
    for p in range(0, N, ranks):
        found_true = False
        for k in range(0, K):
            if itervals[p + k][label_idx] == 1:
                tp += 1.0
                found_true = True
                break
        if not found_true:
            fn += 1.0

    return tp / (tp + fn)


def precision_at_1(predictions, K=1, ranks=10):
    """Precision = TP/(TP+FP)"""
    tp, fp = 0, 0
    itervals = predictions.values
    label_idx = predictions.columns.get_loc("label")
    N = itervals.shape[0]
    for p in range(0, N, ranks):
        found_true = False
        for k in range(0, K):
            if itervals[p + k][label_idx] == 1:
                tp += 1.0
                found_true = True
                break
        if not found_true:
            fp += 1.0

    return tp / (tp + fp)


def mean_reciprocal_rank(predictions, ranks=10):
    """MMR = (1/Q)*SUM(1/rank_i)"""
    itervals = predictions.values
    label_idx = predictions.columns.get_loc("label")
    N = itervals.shape[0]
    Q = N / ranks
    reciprical_rank = 0.0
    for p in range(0, N, ranks):
        for rank in range(0, ranks):
            if itervals[p + rank][label_idx] == 1:
                reciprical_rank += (1.0 / (rank + 1.0))
    return reciprical_rank / Q


def main():
    """
    Order of operations:
        1. append column of labels to predictions
            s.t. predictions = <context, response, likelihood> AND <label> ==> labeled_predictions = <context, response, likelihood, label>
        2. rank responses
            s.t. ranked_labeled_predictions = [<context_1, response_1a, likelihood_1a, label_1a>, <context_1, response_1b, likelihood_1b, label_1b>, ..., <context_N, response_Nj, likelihood_Nj, label_1j>]
        3. evaluate predictions
            ex. recall_at_k(labeled_ranked_preditions, 10)
            ex. mean_reciprical_rank(labeled_ranked_predictions)

    for the Ubuntu set, ranks should always be 10 so it's ok to always use the default value
    'predictions' is expecting = <context, response, likelihood, label>
    """
    predictions = load.load_csv("predictions.csv")
    labels = load.load_csv("ubuntu_test.csv")
    ranked_labeled_predictions = rank_predictions(predictions)
    predictions_col = ranked_labeled_predictions['predictions'].values

    R_at_1 = recall_at_k(predictions_col, 1)
    R_at_3 = recall_at_k(predictions_col, 3)
    R_at_5 = recall_at_k(predictions_col, 5)

    MRR = mean_reciprocal_rank(predictions)

    with open('results.txt', 'w') as f:
        print('R@1: {0}\n R@3: {1} \n R@1: {2}\n MAP: {3} \n MRR: {4}'.format(R_at_1, R_at_3, R_at_5, MAP, MRR), file=f)


if __name__ == '__main__':

    print("Loading test file...")
    with open("dumps/test_expanded.pkl", "rb") as f:
        test_set = pickle.load(f)
    print("Done.")

    print("Loading pretrained word embeddings...")
    pretrained_glove = load.load_pretrained_glove("data/glove_filtered.txt", "dumps/glove_pretrained_dict.pkl")
    embedding_mat, word_indices = we.load_embedding_matrix(pretrained_glove, 300)
    print("Done.\n")

    print("Instanciating classifier.")
    classifier = model.modelClassifier(embedding_mat)
    classifier.restore(False)
    print("Done.\n")

    print("Converting datasets into indices")
    model.get_embed_indices(test_set, word_indices, classifier.sequence_length)
    # We also append the index for unique contexts
    indices = indices_gen(len(test_set))
    print("Done.")

    print("Training")
    hypotheses, probs, cost = classifier.classify(test_set)
    print("Done.")

    print("Calculating metrics")
    y_true = np.array(test_set)
    print(y_true.shape)

    sorted_res = join_rank_predictions(y_true, probs, indices)

    print("Step : {}".format(classifier.sess.run(classifier.global_step)))
    for i in [1, 2, 5]:
        print("Recall @ {}: {}".format(i, recall_at_k(sorted_res, i, 10)))

    print("Mean Reciprocal Rank: {}".format(mean_reciprocal_rank(sorted_res, 10)))

    #main()
