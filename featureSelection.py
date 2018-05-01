from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from util import findNearestHitMiss

from sklearn.metrics import confusion_matrix
# from pandas_ml import ConfusionMatrix
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from random import randint

def sfs(x:pd.DataFrame, y:pd.DataFrame, k, clf, score):
    """
    :param x: feature set to be trained using clf. list of lists.
    :param y: labels corresponding to x. list.
    :param k: number of features to select. int
    :param clf: classifier to be trained on the feature subset.
    :param score: utility function for the algorithm, that receives clf, feature subset and labeles, returns a score.
    :return: list of chosen feature indexes
    """

    # create binary vector for feature selection

    features_select = [False for i in range(len(x.columns))]
    num_features_selected = 0


    while num_features_selected < k:
        max_score = 0
        max_feature = 0
        for i in range(len(features_select)):
            # examine each unselected feature
            if not features_select[i]:
                features_select[i] = True
                current_score = score(clf=clf,examples=x.iloc[:,features_select],classification=y)
                if current_score > max_score:
                    max_score = current_score
                    max_feature = i
                features_select[i] = False

        # add the best feature
        features_select[max_feature] = True
        num_features_selected += 1
        print('Accuracy after',num_features_selected,'features is:',max_score)

    return [i for i in range(len(features_select)) if features_select[i]]





# score function for sfs
def scoreForClassfier(clf, examples, classification):
    numOfSplits = 4
    totalAccuracy = 0

    kf = KFold(n_splits=numOfSplits)
    for train_index, valid_index in kf.split(examples):
        # split the data to train set and validation set:
        examples_train, examples_valid = examples.iloc[train_index], examples.iloc[valid_index]
        classification_train, classification_valid = classification.iloc[train_index], classification.iloc[valid_index]

        # train the knn on train set
        clf.fit(examples_train, classification_train)
        # test the classfier on validation set
        totalAccuracy += accuracy_score(classification_valid, clf.predict(examples_valid))

    totalAccuracy = totalAccuracy / numOfSplits
    return totalAccuracy




def reliefFeatureSelection(X:pd.DataFrame,Y:pd.DataFrame,numOfRowsToSample=5):
    """
    Relief algorithm, best accept normalized data
    params: X- copy of DataFrame w/o labels, Y- labels , numOfFeaturesToSelect-int between 1 to num of
            numOfRowsToSample- int T in pesudo-code, number of times to sample rows from data
    Return: sorted list of tuples (feature_name,feature_score) at size numOfFeaturesToSelect
    """
    totalNumOfFeatures = X.select_dtypes(include=[np.number]).shape[1]
    numOfRows = X.shape[0]
    resW = np.zeros(totalNumOfFeatures,dtype=float)
    resW = resW.reshape((1,-1))
    for i in range(numOfRowsToSample):
        currIndex = randint(0, numOfRows - 1)
        nearestHit = findNearestHitMiss(X, Y, currIndex, 'h')
        nearestMiss = findNearestHitMiss(X, Y, currIndex, 'm')
        nearestHit_values = X.loc[[nearestHit]].select_dtypes(include=[np.number]).values
        nearestMiss_values = X.loc[[nearestMiss]].select_dtypes(include=[np.number]).values
        curr_values = X.iloc[[currIndex]].select_dtypes(include=[np.number]).values
        resW += (curr_values - nearestHit_values)**2 - (curr_values - nearestMiss_values)**2

    # print(resW)
    resWMap = list(zip(X.select_dtypes(include=[np.number]).columns,*resW))
    # print(resWMap)
    return sorted(resWMap,key=itemgetter(1),reverse=True)




def embbdedDecisionTree(X:pd.DataFrame,Y:pd.DataFrame,numOfSplits=4,numOfFeaturesToSelect=10):
    totalAccuracy = 0
    numOflabels = Y['Vote'].nunique()
    totalConfusion = np.zeros((numOflabels, numOflabels))
    X = X.select_dtypes(include=[np.number])
    partiesLabels = Y['Vote'].unique()
    # run kfold on trees
    kf = KFold(n_splits=numOfSplits, shuffle=True)
    for train_index, test_index in kf.split(X):
        # split the data to train set and validation set:
        features_train, features_test = X.iloc[train_index], X.iloc[test_index]
        classification_train, classification_test = Y.iloc[train_index], Y.iloc[test_index]

        # train the tree on train set
        estimator = tree.DecisionTreeClassifier(criterion="entropy")
        estimator.fit(features_train, classification_train)
        # test the tree on validation set
        totalAccuracy += accuracy_score(classification_test, estimator.predict(features_test))
        totalConfusion += confusion_matrix(classification_test.values, estimator.predict(features_test),labels=partiesLabels)

    # calculate accuracy and confusion matrix
    totalAccuracy = totalAccuracy / numOfSplits
    totalConfusion = np.rint(totalConfusion).astype(int)

    print('Total Accuracy of tree is:',totalAccuracy)
    # print('Confusion Matrix of tree is:\n',totalConfusion)
    resWMap = list(zip(X.select_dtypes(include=[np.number]).columns, (estimator.feature_importances_)))
    resWMap = sorted(resWMap,key=itemgetter(1),reverse=True)
    # print(resWMap)
    return resWMap,totalConfusion


