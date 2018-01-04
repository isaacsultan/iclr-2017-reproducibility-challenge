
# This code was adapted from https://github.com/nyu-mll/multiNLI
# See NOTICE.txt for modification details


def evaluate_classifier(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    hypotheses, probs, cost = classifier(eval_set)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        if hypothesis == eval_set[i][2]:
            correct += 1        
    return correct / float(len(eval_set)), cost


def evaluate_final(restore, classifier, eval_set, batch_size):
    """
    Function to get percentage accuracy of the model, evaluated on a set of chosen datasets.
    
    restore: a function to restore a stored checkpoint
    classifier: the model's classfier, it should return logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    restore(best=True)
    percentages = []

    hypotheses, probs, cost = classifier(eval_set)
    correct = 0
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        # Column indexed 2 is the label (0.0 or 1.0)
        if hypothesis == eval_set[i][2]:    
            correct += 1      
    return correct / float(len(eval_set))