import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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


def pcaTrain(x_data:pd.DataFrame, y_data:pd.DataFrame):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    #normalize the data
    x_data_norm = StandardScaler().fit_transform(x_data)
    print(x_data_norm.head(10))
    #initiate PCA
    pca = PCA(n_components=2) #for visualizaion 
    pca_res = pca.fit_transform(x_data_norm)

    pca_df = pd.DataFrame(data=pca_res,columns=['principal component 1', 'principal component 2'])
    print(pca_df.head(10))
    final_df = pd.concat([pca_df,y_data],axis=1)
    labels = y_data.Vote.unique()
    # plt.scatter()
    
    
    
    
def fillNAByLabelMode(X:pd.DataFrame,Y:pd.DataFrame,index):
    if X.index.dtype == 'float':
        print('ERROR needs to be a discrete category')
    df = X
    df['Vote'] = Y.copy().values
    partyList = df['Vote'].unique()
    df[index + 'FillByMode'] = df[index]
    for p in partyList:
        mask = df.Vote == p
        colByLabel = df[mask]
        currMode = colByLabel[index].mode().iloc[0] # just the first mode, could be more than 1
        print('party',p,'mode is:',currMode) # TODO remove
        # df.loc[df[df[mask][index].isnull()],index + 'FillByMode'] = currMode
        # df[mask][index] = df[mask][index].fillna(currMode)
        df.loc[(mask) & (df[index + 'FillByMode'].isnull()),index + 'FillByMode'] = currMode
    return df.drop('Vote', axis=1)


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
        curr = colByLabel[index].mean if meanOrMedian == 'Mean' else colByLabel[index].median
        df.loc[(mask) & (df[newColName].isnull()),newColName] = curr
    return df.drop('Vote', axis=1)




df['IncomeMinusExpenses'] = df.Yearly_IncomeK - df.Yearly_ExpensesK