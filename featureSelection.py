import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from util import findNearestHitMiss

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
                current_score = score(classifier=clf,examples=x[:,features_select],classification=y)
                if current_score > max_score:
                    max_score = current_score
                    max_feature = i
                features_select[i] = False

        # add the best feature
        features_select[max_feature] = True
        num_features_selected += 1

    return [i for i in range(len(features_select)) if features_select[i]]





# score function for sfs
def scoreForClassfier(clf, examples, classification):
    numOfSplits = 4
    totalAccuracy = 0

    kf = KFold(n_splits=numOfSplits)
    for train_index, valid_index in kf.split(examples):
        # split the data to train set and validation set:
        examples_train, examples_valid = examples[train_index], examples[valid_index]
        classification_train, classification_valid = classification[train_index], classification[valid_index]

        # train the knn on train set
        clf.fit(examples_train, classification_train)
        # test the classfier on validation set
        totalAccuracy += accuracy_score(classification_valid, clf.predict(examples_valid))

    totalAccuracy = totalAccuracy / numOfSplits
    return totalAccuracy



def reliefFeatureSelection(X:pd.DataFrame,Y:pd.DataFrame,numOfFeaturesToSelect=20,numOfRowsToSample=5):
    totalNumOfFeatures = X.select_dtypes(include=[np.number]).shape[1]
    numOfRows = X.shape[0]
    resW = np.zeros(totalNumOfFeatures,dtype=float)
    resW = resW.reshape((1,-1))
    for i in range(numOfRowsToSample):
        currIndex = randint(0, numOfRows)
        nearestHit = findNearestHitMiss(X, Y, currIndex, 'h')
        nearestMiss = findNearestHitMiss(X, Y, currIndex, 'm')
        nearestHit_values = X.loc[[nearestHit]].select_dtypes(include=[np.number]).values
        nearestMiss_values = X.loc[[nearestMiss]].select_dtypes(include=[np.number]).values
        curr_values = X.iloc[[currIndex]].select_dtypes(include=[np.number]).values
        resW += (curr_values - nearestHit_values)**2 - (curr_values - nearestMiss_values)**2

    print(resW)
    resWMap = list(zip(X.select_dtypes(include=[np.number]).columns,*resW))
    print(resWMap)
    return sorted(resWMap[:numOfFeaturesToSelect],key=lambda x:x[1],reverse=True)
