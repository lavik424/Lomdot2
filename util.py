import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from random import randint



colPool = [ '#bd2309', '#bbb12d', '#1480fa', '#14fa2f', '#000000',\
                     '#faf214', '#2edfea', '#ea2ec4', '#ea2e40', '#cdcdcd',\
                    '#577a4d', '#2e46c0', '#f59422', '#219774', '#8086d9' ]

colToInt = pd.Index(['Occupation_Satisfaction', 'Last_school_grades',\
            'Number_of_differnt_parties_voted_for','Number_of_valued_Kneset_members',\
            'Num_of_kids_born_last_10_years'])

### Print plot from training set on category dtype###
def describeAndPlot(df:pd.DataFrame):
    # df.describe()

    #for categorical columns
    catFeat = df.keys()[df.dtypes.map(lambda x: x!=np.number)]
    catFeat = catFeat.drop('Vote')
    catFeat = catFeat.union(colToInt)
    for key in catFeat:
        new_plot = pd.crosstab([df.Vote], df[key])
        new_plot.plot(kind='bar', stacked=True,\
                      color=colPool, grid=False)
        title = "Distribution of {} in different parties"
        plt.title(title.format(key))
        plt.xlabel('Name of Party')
        plt.ylabel('Number of Voters')
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        title += '.png'
        plotName = './plots/' + title.format(key)
        plt.savefig(plotName,bbox_inches="tight")
        plt.clf()

    # for numeric columns
    numFeat = df.keys()[df.dtypes.map(lambda x: x == np.number)]
    numFeat = numFeat.difference(colToInt)
    partyMap = {p:i for i,p in enumerate(df['Vote'].unique())}
    indexList = [i for i in partyMap.values()]
    partyList = [p for p in partyMap]

    for key in numFeat:
        rows = df[key].notnull()
        x = df.loc[rows,key]
        y = df.loc[rows,'Vote']
        y = y.map(partyMap)

        plt.scatter(x,y)
        title = "Scatter plot of {} in different parties"
        plt.title(title.format(key))
        plt.xlabel('TBD')
        plt.xlim(np.floor(np.min(x)),np.ceil(np.max(x)))
        plt.ylabel('Name of Party')
        plt.yticks(indexList,partyList)
        title += '.png'
        plotName = './plots/' + title.format(key)
        plt.savefig(plotName, bbox_inches="tight")
        plt.clf()



### Creates HIST plots for numerical categories ###

def histForFloat(df:pd.DataFrame):
    numFeat = df.keys()[df.dtypes.map(lambda x: x == np.number)]
    numFeat = numFeat.difference(colToInt)
    partyMap = {p:i for i,p in enumerate(df['Vote'].unique())}


    for key in numFeat:
        partyList = df['Vote'].unique()
        plt.figure(figsize=(40,30))
        mainTitle = "Hist plots of {}"
        plt.suptitle(mainTitle.format(key))
        rows = df[key].notnull()
        maxXValue = np.ceil(np.max(df[key]))    ## to assure all subplots will have same x scale
        minXValue = np.floor(np.min(df[key]))   ## same as above
        for i,p in enumerate(partyList):
            mask = df.Vote == p
            x = df.loc[mask & rows,key]
            plt.subplot(3,4,i+1)
            n,bins,patches = plt.hist(x=x,bins=20)
            plt.title(p)
            plt.ylabel('Number of Voters')
            plt.xlim(minXValue, maxXValue)
            plt.ylim(0,1+np.max(n).astype(int))
            ## Trying to add line for normal distribution
            # mu = x.mean()
            # sigma = np.std(x.values)
            # print('mean is:',mu,'std is:',sigma)
            # normDis = np.linspace(np.floor(np.min(x)), np.ceil(np.max(x)), bins.shape[0])
            # y = norm.pdf(bins, mu, sigma)
            # plt.plot(bins, y, 'r--')
            plt.plot(bins)

        mainTitle += '.png'
        plotName = './plots/' + mainTitle.format(key)
        plt.savefig(plotName, bbox_inches="tight")
        plt.close()

    
### Function for TRAIN DATA that fill nan cells in object categories with mode value ###
def fillNAByLabelMode(X:pd.DataFrame,Y:pd.DataFrame,index):
    if X.index.dtype == 'float':
        print('ERROR needs to be a discrete category')
    df = X
    df['Vote'] = Y.copy().values
    # df.loc[:, 'Vote'] = Y['Vote']
    partyList = df['Vote'].unique()
    df[index + 'FillByMode'] = df[index]
    for p in partyList:
        mask = df.Vote == p
        colByLabel = df[mask]
        currMode = colByLabel[index].mode().iloc[0] # just the first mode, could be more than 1
        # print('party',p,'mode is:',currMode) # TODO remove
        # df.loc[df[df[mask][index].isnull()],index + 'FillByMode'] = currMode
        # df[mask][index] = df[mask][index].fillna(currMode)
        df.loc[(mask) & (df[index + 'FillByMode'].isnull()),index + 'FillByMode'] = currMode
    return df.drop('Vote', axis=1)



### Function for TEST/VALIDATION DATA that fill nan cells in object categories with mode value ###
def fillNATestValMode(X:pd.DataFrame,index):
    if X.index.dtype == 'float':
        print('ERROR needs to be a discrete category')
    df = X
    df[index + 'FillByMode'] = df[index]
    currMode = df[index].mode().iloc[0]
    df.loc[(df[index + 'FillByMode'].isnull()), index + 'FillByMode'] = currMode
    return df



### Function for TRAIN DATA that fill nan cells in numeric categories with mean or median value ###
def fillNAByLabelMeanMedian(X:pd.DataFrame,Y:pd.DataFrame,index,meanOrMedian):
    if not meanOrMedian in ('Mean','Median'):
        print('ERROR should state mean or median only')
        return X
    if X.index.dtype == np.number:
        print('ERROR needs to be a numeric category')
        return X
    df = X
    df['Vote'] = Y.copy().values
    partyList = df['Vote'].unique()
    newColName = index + 'FillBy' + meanOrMedian
    df[newColName] = df[index]
    for p in partyList:
        mask = df.Vote == p
        colByLabel = df[mask]
        curr = np.nanmean(colByLabel[index]) if meanOrMedian == 'Mean' else np.nanmedian(colByLabel[index])
        df.loc[(mask) & (df[newColName].isnull()),newColName] = curr
    return df.drop('Vote', axis=1)


### Function for TEST/VALIDATION DATA that fill nan cells in numeric categories with mean or median value ###
def fillNATestValMeanMedian(X:pd.DataFrame,index,meanOrMedian):
    if not meanOrMedian in ('Mean','Median'):
        print('ERROR should state mean or median only')
        return X
    if X.index.dtype == np.number:
        print('ERROR needs to be a numeric category')
        return X
    df = X
    newColName = index + 'FillBy' + meanOrMedian
    df[newColName] = df[index]
    curr = np.nanmean(df[index]) if meanOrMedian == 'Mean' else np.nanmedian(df[index])
    df.loc[(df[newColName].isnull()),newColName] = curr
    return df


def distanceBetween2Samples(sam1,sam2):
    """
    Function that compute the distane between 2 samples from DataFrame. Should get normalized data, w/o nan values
    Let x1,x2,...,xN values of N numeric features of sam1
    and y1,y2,...,yN values of N numeric features of sam2
    Return: sqrt((x1-y1)^2+(x2-y2)^2+...+(xN-yN)^2)
    """
    sam1 = sam1.select_dtypes(include=[np.number]).values
    sam2 = sam2.select_dtypes(include=[np.number]).values
    res = np.sqrt(np.nansum((sam1-sam2)**2))
    return res



def findNearestHitMiss(X:pd.DataFrame,Y:pd.DataFrame,samIndex,hitMiss='h'):
    """
    Finds closet sample to sam in the same/different label. Uses distanceBetween2Samples(), should get normalized data
    params: X- copy of DataFrame w/o labels, Y- labels , samIndex- index of the sample in X with iloc (X relative row's index)
            hitMiss- 'h' for hit(same label), 'm' for miss (closest in other label)
    Return: index of closest sample in the same\other label, original index use with loc
    """
    if hitMiss != 'h' and hitMiss != 'm':
        print('ERROR must state \'h\' for hit or \'m\' for miss')
        return -1
    # merge X+Y
    df = X
    df['Vote'] = Y.values

    sampleToCompare = df.iloc[[samIndex]] 
    realSamIndex = df.iloc[[samIndex]].index[0] # beacuse its easier to iterate over iloc but loc gives exact location
    # print('samIndex=',samIndex,'but real index is:',realSamIndex)

    label = sampleToCompare['Vote'] # gets sam's label
    # print(label)
    label = label.get_values()[0]
    # print('The label is:',label)
    if hitMiss == 'h':
        mask = df.Vote == label
    else:
        mask = df.Vote != label
    rowsByLabel = df[mask]
    minIndex = -1
    minScore = np.inf
    
    for i in range(rowsByLabel.shape[0]): # iterate over rows
        currIndex = rowsByLabel.iloc[[i]].index[0] # gets the index of the row in the original df
        # print(currIndex)
        if realSamIndex == currIndex: 
            continue
        curr = distanceBetween2Samples(sampleToCompare, rowsByLabel.iloc[[i]])
        # print(curr)
        if curr < minScore:
            minScore = curr
            minIndex = currIndex
    return minIndex



def heuristicFindNearestHitMiss(X: pd.DataFrame, Y: pd.DataFrame, samIndex, hitMiss='h'):
    """
    Finds closet sample to sam in the same/different label. Uses distanceBetween2Samples(), should get normalized data
    params: X- copy of DataFrame w/o labels, Y- labels , samIndex- index of the sample in X with iloc (X relative row's index)
            hitMiss- 'h' for hit(same label), 'm' for miss (closest in other label)
    Return: index of closest sample in the same\other label, original index use with loc
    """
    if hitMiss != 'h' and hitMiss != 'm':
        print('ERROR must state \'h\' for hit or \'m\' for miss')
        return -1
    # merge X+Y
    df = X
    df['Vote'] = Y.values

    sampleToCompare = df.iloc[[samIndex]]
    realSamIndex = df.iloc[[samIndex]].index[
        0]  # beacuse its easier to iterate over iloc but loc gives exact location
    # print('samIndex=',samIndex,'but real index is:',realSamIndex)

    label = sampleToCompare['Vote']  # gets sam's label
    # print(label)
    label = label.get_values()[0]
    # print('The label is:',label)
    if hitMiss == 'h':
        mask = df.Vote == label
    else:
        mask = df.Vote != label
    rowsByLabel = df[mask]
    minIndex = -1
    minScore = np.inf
    # print('shape of label=',rowsByLabel.shape[0])
    randArray = np.random.randint(rowsByLabel.shape[0],size=100)
    for i in randArray:  # Sample 100 indices
        # print('index is:',heuristicIndex)
        currIndex = rowsByLabel.iloc[[i]].index[0]  # gets the index of the row in the original df
        # print(currIndex)
        if realSamIndex == currIndex:
            continue
        curr = distanceBetween2Samples(sampleToCompare, rowsByLabel.iloc[[i]])
        # print(curr)
        if curr < minScore:
            minScore = curr
            minIndex = currIndex
    return minIndex

        # return np.min(np.vectorize(\
    #     lambda row:distanceBetween2Samples(df.iloc[[samIndex]],row)(rowsByLabel)))
    #         # if row.index != samIndex else np.inf)(rowsByLabel)))


def fillNanWithOtherColumns(X:pd.DataFrame,Y:pd.DataFrame,listOfColsWithConnection):
    col2edit = X[listOfColsWithConnection]
    # for col in listOfColsWithConnection:
    for i in np.arange(col2edit.shape[0]):
        if i % 100 == 0:
            print(i) # TODO remove
        counter = 0
        while col2edit.iloc[i].hasnans and counter < 3:
            # nearestHit = findNearestHitMiss(col2edit,Y,i,'h') # slower iterate over all the data
            # print(col2edit.shape[0]) # TODO remove
            nearestHit = heuristicFindNearestHitMiss(col2edit, Y, i, 'h') # faster
            if nearestHit != -1:
                goodSample = col2edit.loc[nearestHit]
                col2edit.iloc[i] = col2edit.iloc[i].fillna(goodSample)
                goodSample = goodSample.fillna(col2edit.iloc[i])
            counter += 1
    return col2edit



def changeOutlierToMean(X:pd.DataFrame,Y:pd.DataFrame,index,label,lowerBound,upperBound):
    # merge X+Y
    df = X
    df['Vote'] = Y.values
    mask = df.Vote == label
    rowsByLabel = df[mask]
    meanValue = np.nanmean(rowsByLabel[index])
    # print('mean of',index,'is:',meanValue)
    if lowerBound != None:
        df.loc[(mask) & (df[index] < lowerBound),index] = meanValue
    if upperBound != None:
        df.loc[(mask) & (df[index] > upperBound),index] = meanValue
    return df.drop('Vote', axis=1)