import pandas as pd
import numpy as np

from operator import itemgetter
from util import findNearestHitMiss
from random import randint



def reliefFeatureSelection(X:pd.DataFrame,Y:pd.DataFrame,numOfRowsToSample=5):
    """
    Relief algorithm, best accept normalized data
    params: X- copy of DataFrame w/o labels, Y- labels , numOfFeaturesToSelect-int between 1 to num of
            numOfRowsToSample- int T in pseudo-code, number of times to sample rows from data
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
        resW += (curr_values - nearestMiss_values)**2 - (curr_values - nearestHit_values)**2

    # print(resW)
    resWMap = list(zip(X.select_dtypes(include=[np.number]).columns,*resW))
    # print(resWMap)
    return sorted(resWMap,key=itemgetter(1),reverse=True)
