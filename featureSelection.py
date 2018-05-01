

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


from operator import itemgetter
from sklearn.metrics import confusion_matrix
# from pandas_ml import ConfusionMatrix
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score





# score function for sfs
def scoreForClassfier(clf, examples, classification):
    numOfSplits = 4
    totalAccuracy = 0
    # numOflabels = classification['Vote'].nunique()
    # totalConfusion = np.zeros((numOflabels, numOflabels))
    # partiesLabels = classification['Vote'].unique()
    kf = KFold(n_splits=numOfSplits)
    for train_index, valid_index in kf.split(examples):
        # split the data to train set and validation set:
        examples_train, examples_valid = examples.iloc[train_index], examples.iloc[valid_index]
        classification_train, classification_valid = classification.iloc[train_index], classification.iloc[valid_index]

        # train the knn on train set
        clf.fit(examples_train, classification_train)
        # test the classfier on validation set
        totalAccuracy += accuracy_score(classification_valid, clf.predict(examples_valid))
        # totalConfusion += confusion_matrix(classification_valid.values, clf.predict(examples_valid),
        #                                    labels=partiesLabels)

    totalAccuracy = totalAccuracy / numOfSplits
    return totalAccuracy




def embeddedDecisionTree(X:pd.DataFrame,Y:pd.DataFrame,numOfSplits=4,numOfFeaturesToSelect=10):
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


