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

def pearsonCorrelation(col1:pd.Series, col2:pd.Series):
    """
    calculates pearson correlation between to columns
    :param col1:
    :param col2:
    :return:
    """
    col1array = col1.values()
    col2array = col2.values()

    return sp.stats.pearsonr(col1array, col2array)

def mutualInformation(label:pd.Series, x:pd.Series):
    """
    calculates pearson correlation between to columns
    :param label:
    :param x:
    :return:
    """

    col1array = label.values
    col2array = x.values

    return metrics.mutual_info_score(label,x)

def plotPCA(X:pd.DataFrame, Y:pd.Series, title="PCA of features"):

    Y = Y.values
    X = X.values

    X = MinMaxScaler().fit_transform(X)
    # X = StandardScaler().fit_transform(X)
    print(PCA().fit(X).singular_values_)
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



# Not working 
### Normalize one column (index) in a df.
### index = col name , method = MinMax for minMaxScaler or Standard for StandardScaler
# def scaleSingleColumn(df: pd.DataFrame, index, method='MinMax'):
#     if method not in ('MinMax', 'Standard'):
#         print('ERROR should state MinMax or Standard only')
#         return df
#     if index not in df.columns:
#         print('ERROR index have to be valid column name in df')
#         return df
#     if method == 'MinMax':
#         scaler = MinMaxScaler() if method == 'MinMax' else StandardScaler()
#
#     print('Before scaler:\n')#,df.index.describe())
#     print(df[index].head(5))
#     df.loc[:,df[index]] = scaler.fit_transform(df.index)
#     print('After scaler:\n')#,df.index.describe())
#     print(df[index].head(5))
#     return df



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




### Normalize by Normal Standard Distribution (T test??) one column (index) in a df.
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