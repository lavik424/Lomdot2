import pandas as pd
import numpy as np
import scipy as sp
from sklearn import metrics

import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

def pearsonCorrelation(col1:np.ndarray, col2:np.ndarray):
    """
    calculates pearson correlation between to columns
    :param col1:
    :param col2:
    :return:
    """

    return sp.stats.pearsonr(col1, col2)

def mutualInformation(label:np.ndarray, x:np.ndarray):
    """
    calculates pearson correlation between to columns
    :param label:
    :param x:
    :return:
    """


    if len(label.shape) == 2:
        label = label.flatten()
    if len(x.shape) == 2:
        x = x.flatten()

    return metrics.mutual_info_score(label,x)

def plotPCA(X:pd.DataFrame, Y:pd.Series, title="PCA of features", normalize = "min-max"):

    Y = Y.values
    X = X.values

    if normalize == "min-max":
        X = MinMaxScaler().fit_transform(X)
    elif normalize == "normal":
        X = StandardScaler().fit_transform(X)
    PCAll = PCA().fit(X)
    print(PCAll.explained_variance_ratio_)

    # u, s, vh = np.linalg.svd(X, full_matrices=False)

    X = PCA(n_components=2).fit_transform(X)

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    plt.title(title)

    labels_with_colors = {"Blues" : "b",
                          "Browns" : "brown",
                          "Purples": "purple",
                          "Whites" : "k",
                          "Pinks": "pink",
                          "Turquoises": "c",
                          "Oranges": "orange",
                          "Yellows": "yellow",
                          "Greens": "g",
                          "Greys": "grey",
                          "Reds": "r"}

    for label, color in labels_with_colors.items():
        labelIndexes = np.where(Y == label)
        # plt.subplot(3,4,i)
        plt.scatter(X[labelIndexes, 0], X[labelIndexes, 1], c=color,
                    label=label)
        # plt.scatter(X[one_class, 0], X[one_class, 1], s=80, c='orange',
        #         label='Class 2')

    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    # plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    # if subplot == 2:
    #     plt.xlabel('First principal component')
    #     plt.ylabel('Second principal component')
    #     plt.legend(loc="upper left")
    plt.show()



### Normalize by MinMax one column (index) in a df.
### index = col name
### Return: updated df
def scaleMinMaxSingleColumn(df: pd.DataFrame, index):
    if index not in df.columns:
        print('ERROR index have to be valid column name in df')
        return df
    # print('Before scaler:\n')  # ,df.index.describe())
    # print(df[index].head(5))
    max_value = np.max(df[index])
    min_value = np.min(df[index])
    df[index] = (df[index] - min_value) / (max_value - min_value)
    # print('After scaler:\n')  # ,df.index.describe())
    # print(df[index].head(5))
    return df




### Normalize by Normal Standard Distribution (Z/T test??) one column (index) in a df.
### index = col name
### Return: updated df
def scaleNormalSingleColumn(df: pd.DataFrame, index):
    if index not in df.columns:
        print('ERROR index have to be valid column name in df')
        return df
    # print('Before scaler:\n')  # ,df.index.describe())
    # print(df[index].head(5))
    mean_value = np.mean(df[index])
    std_value = np.std(df[index])
    df[index] = (df[index] - mean_value) / (std_value)
    # print('After scaler:\n')  # ,df.index.describe())
    # print(df[index].head(5))
    return df


def plotColvsCol(X1:pd.Series, X2:pd.Series, Y:pd.Series, title="PCA of features", normalize ="min-max"):

    if normalize == "min-max":
        X1 = MinMaxScaler().fit_transform(X1)
        X2 = MinMaxScaler().fit_transform(X2)
    elif normalize == "normal":
        X1 = StandardScaler().fit_transform(X1)
        X2 = StandardScaler().fit_transform(X2)

    min_x = np.min(X1)
    max_x = np.max(X1)

    min_y = np.min(X2)
    max_y = np.max(X2)

    plt.title(title)

    labels_with_colors = {"Blues" : "b",
                          "Browns" : "brown",
                          "Purples": "purple",
                          "Whites" : "k",
                          "Pinks": "pink",
                          "Turquoises": "c",
                          "Oranges": "orange",
                          "Yellows": "yellow",
                          "Greens": "g",
                          "Greys": "grey",
                          "Reds": "r"}

    pointSize = 90
    for label, color in labels_with_colors.items():
        labelIndexes = np.where(Y == label)
        # plt.subplot(3,4,i)
        plt.scatter(X1[labelIndexes], X2[labelIndexes], c=color,
                    label=label, s= pointSize)
        pointSize -= 7
        # plt.scatter(X[one_class, 0], X[one_class, 1], s=80, c='orange',
        #         label='Class 2')

    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    # plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    # if subplot == 2:
    #     plt.xlabel('First principal component')
    #     plt.ylabel('Second principal component')
    #     plt.legend(loc="upper left")
    mainTitle = title + '.png'
    plotName = './plots/colvscol/' + mainTitle
    plt.savefig(plotName)
    plt.close()

