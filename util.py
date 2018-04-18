import numpy as np
import pandas as pd




colPool = [ '#bd2309', '#bbb12d', '#1480fa', '#14fa2f', '#000000',\
                     '#faf214', '#2edfea', '#ea2ec4', '#ea2e40', '#cdcdcd',\
                    '#577a4d', '#2e46c0', '#f59422', '#219774', '#8086d9' ]

colToInt = pd.Index(['Occupation_Satisfaction', 'Last_school_grades',\
            'Number_of_differnt_parties_voted_for','Number_of_valued_Kneset_members',\
            'Num_of_kids_born_last_10_years'])

### Print plot from training set on category dtype###
def describeAndPlot(df:pd.DataFrame):
    import matplotlib.pyplot as plt
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
        plt.savefig(title.format(key),bbox_inches="tight")
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
        # print(key,':    max=',np.max(x),'   min=',np.min(x))
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
        plt.savefig(title.format(key), bbox_inches="tight")
        plt.clf()


def pcaTrain(df:pd.DataFrame):
    pass