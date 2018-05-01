import pandas as pd
import numpy as np


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
    orderOfSelect = []

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
        orderOfSelect.append(max_feature)
        num_features_selected += 1
        print('Accuracy after',num_features_selected,'features is:',max_score)

    # return [i for i in range(len(features_select)) if features_select[i]]
    return orderOfSelect