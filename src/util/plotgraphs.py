import pickle

import pandas as pd
from scipy.interpolate import spline
import numpy as np
from matplotlib import pyplot as plt



def show_metric_from_csv(file):
    data = pd.read_csv(file, encoding="utf-8", keep_default_na=False, header=0).values

    # Recall @ 1
    plt.plot(data[:,0], data[:,1], "g-")
    plt.xlabel("Global step")
    plt.show()

    # Recall @ 2
    plt.plot(data[:,0], data[:, 2], "r-")
    plt.xlabel("Global step")
    plt.show()

    # Recall @ 5
    plt.plot(data[:,0], data[:,3], "b-")
    plt.xlabel("Global step")
    plt.show()

    # Mean reciprocal rank
    plt.plot(data[:,0], data[:, 4], "k-")
    plt.xlabel("Global step")
    plt.show()
    


def fix_initial_steps(data):
    """The first 39 indices used the step variable instead of the global_step.
    We therefore update these indices to match the next values"""
    
    index = 5402

    for i in range(38, -1, -1):
        data[i][0] = index
        index -= 50

    return data

def show_accuracy_graph(data, smooth=False):
    # Indices 1 and 2 refers to val. acc. and val. loss
    # Indices 3 and 4 refers to train. acc. and train. cost

    if smooth:
        newdata = np.zeros((300, 5))
        newdata[:, 0] = np.linspace(data[:,0].min(), data[:,0].max(), 300)

        # Smooth out the data
        for i in range(1, 5):
            newdata[:,i] = spline(data[:, 0], data[:, i], newdata[:,0])

        # Replace actual data by smooth data
        data = newdata

    # Accuracy
    plt.plot(data[:,0], data[:,1], "g-", data[:,0], data[:, 3], "r-")
    #plt.title("Accuracy over training")
    plt.ylim([0.65, 0.95])
    plt.xlabel("Global step")
    #plt.ylabel("Accuracy")
    plt.legend(["Validation set", "Mini-training set"])
    plt.show()

    # Loss
    plt.plot(data[:,0], data[:,2], "g-", data[:,0], data[:, 4], "r-")
    #plt.title("Loss")
    plt.ylim([0, 0.008])
    plt.xlabel("Global step")
    #plt.ylabel("Loss")
    plt.legend(["Validation set", "Mini-training set"])
    plt.show()


if __name__ == "__main__":
    with open("checkpoints/esim.pkl", "rb") as f:
        data = pickle.load(f)

    data = fix_initial_steps(data)
    show_accuracy_graph(np.array(data))

    # Then show the metric graph
    show_metric_from_csv("results/ESIM Results.csv")