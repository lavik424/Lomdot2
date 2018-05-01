
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from util import *
import stats
from featureSelection import *

from sfs import sfs
from relief import reliefFeatureSelection


def add_missing_dummy_columns( d, columns ,original_col_name ):
    missing_cols = set( columns ) - set( d.columns ) - set(original_col_name)
    for c in missing_cols:
        d[c] = 0

def fix_multivar_columns_for_test(data_test, data_train, original_col_name):
    """
    modify multicategorial columns of test set to fit the categories exists in train set
    :param data_test: DataFrame of testset (before get_dummies())
    :param data_train: dataframe of trainset (after get_dummies but include original columns)
    :param original_col_name: the columns that are categorial and contains multivalues
    :return: new test set with categorial column splitted according to values in train set
    """

    for f in original_col_name:
        data_test[f] = data_test[f].astype("category")
        if not data_test[f].cat.categories.isin(data_train[f].cat.categories).all():
            data_test.loc[~data_test[f].isin(data_train[f].cat.categories), f] = np.nan

    data_test = pd.get_dummies(data_test, columns=original_col_name, dummy_na=True)

    add_missing_dummy_columns(data_test, data_train.columns, original_col_name)

    # make sure we have all the columns we need
    assert(set(data_test) - set(data_train.columns) == set())

    extra_cols = set(data_test.columns) - set(data_train.columns)
    assert (not extra_cols)

    # d = d[ columns ]
    return data_test


def setTypesToCols(trainX:pd.DataFrame, trainY:pd.DataFrame,
                   validX:pd.DataFrame, validY: pd.DataFrame,
                   testX: pd.DataFrame, testY: pd.DataFrame):

    colOfMultiCategorial = ["Most_Important_IssueFillByMode", "Will_vote_only_large_partyFillByMode",
                            "Main_transportationFillByMode", "OccupationFillByMode"]

    colOfOrderedCategorial = ["Age_groupFillByMode"]

    colOfBinaryCategorial = [c for c in trainX.keys()[trainX.dtypes.map(lambda x: x=='object')]
                       if c not in colOfMultiCategorial and c not in colOfOrderedCategorial]

    ### translate ordered categorial
    f = colOfOrderedCategorial[0]
    # categorize train set
    trainX[f] = trainX[f].astype("category")
    trainX[f + "Int"] = trainX[f].cat.rename_categories(
        {'Below_30': 0, '30-45': 1, '45_and_up': 2}).astype(float)
    trainX.loc[trainX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

    # Let's creat a crosstabcross-tabulation to look at this transformation
    pd.crosstab(trainX[f+"Int"], trainX[f], rownames=[f+"Int"], colnames=[f])

    # categorize valid set
    validX[f] = validX[f].astype("category")
    if validX[f].cat.categories.isin(validX[f].cat.categories).all():
        validX[f] = validX[f].cat.rename_categories(validX[f].cat.categories)
    else:
        print("\n\nTrain and Valid don't share the same set of categories in feature '", f, "'")
    # legitIndex = trainX[f].notnull()
    validX[f + "Int"] = validX[f].cat.rename_categories(
        {'Below_30': 0, '30-45': 1, '45_and_up': 2}).astype(float)
    validX.loc[validX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

    # categorize test set
    testX[f] = testX[f].astype("category")
    if testX[f].cat.categories.isin(testX[f].cat.categories).all():
        testX[f] = testX[f].cat.rename_categories(testX[f].cat.categories)
    else:
        print("\n\nTrain and Test don't share the same set of categories in feature '", f, "'")
    testX[f + "Int"] = testX[f].cat.rename_categories(
        {'Below_30': 0, '30-45': 1, '45_and_up': 2}).astype(float)
    testX.loc[testX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

    ### translate binary categorial
    for f in colOfBinaryCategorial:
        # categorize train set
        trainX[f] = trainX[f].astype("category")
        trainX[f + "Int"] = trainX[f].cat.rename_categories(range(trainX[f].nunique())).astype(int)
        trainX.loc[trainX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

        # Let's creat a crosstabcross-tabulation to look at this transformation
        # pd.crosstab(train[f+"Int"], train[f], rownames=[f+"Int"], colnames=[f])

        # categorize valid set
        validX[f] = validX[f].astype("category")
        if validX[f].cat.categories.isin(validX[f].cat.categories).all():
            validX[f] = validX[f].cat.rename_categories(validX[f].cat.categories)
        else:
            print("\n\nTrain and Valid don't share the same set of categories in feature '", f, "'")
        validX[f + "Int"] = validX[f].cat.rename_categories(range(validX[f].nunique())).astype(int)
        validX.loc[validX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

        # categorize test set
        testX[f] = testX[f].astype("category")
        if testX[f].cat.categories.isin(testX[f].cat.categories).all():
            testX[f] = testX[f].cat.rename_categories(testX[f].cat.categories)
        else:
            print("\n\nTrain and Test don't share the same set of categories in feature '", f, "'")
        testX[f + "Int"] = testX[f].cat.rename_categories(range(testX[f].nunique())).astype(int)
        testX.loc[testX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

    ### translate multivar categorial
    for f in colOfMultiCategorial:
        # categorize train set
        trainX[f] = trainX[f].astype("category")
    trainX = pd.concat([pd.get_dummies(trainX, columns=colOfMultiCategorial, dummy_na=True),
                        trainX[colOfMultiCategorial]], axis=1)

    testX = pd.concat([fix_multivar_columns_for_test(testX,trainX,colOfMultiCategorial),
                       testX[colOfMultiCategorial]], axis=1)

    validX = pd.concat([fix_multivar_columns_for_test(validX,trainX,colOfMultiCategorial),
                        trainX[colOfMultiCategorial]], axis=1)

    return trainX, trainY, validX, validY, testX, testY

def creatColVsColCorrelationMatrix(x_train_cat_number_only):
    colvscolresults = []
    colList = x_train_cat_number_only.columns
    for i in range(len(colList)):
        for j in range(i+1,len(colList)):
            c1 = colList[i]
            c2 = colList[j]
            if c1 != 'Vote' and c2 != 'Vote':
                c1_scaled = MinMaxScaler().fit_transform(x_train_cat_number_only[c1].reshape(-1, 1))
                c2_scaled = MinMaxScaler().fit_transform(x_train_cat_number_only[c2].reshape(-1, 1))
                mi = stats.mutualInformation(c1_scaled, c2_scaled)
                pearson = stats.pearsonCorrelation(c1_scaled, c2_scaled)

                colvscolresults.append((c1 + " VS " + c2, mi, pearson))

    return colvscolresults


def drawColVsColScatterPlot(x_train_cat_number_only, Y):
    colList = x_train_cat_number_only.columns
    for i in range(len(colList)):
        for j in range(i+1,len(colList)):
            c1 = colList[i]
            c2 = colList[j]
            if c1 != 'Vote' and c2 != 'Vote':
                stats.plotColvsCol(x_train_cat_number_only[c1].reshape(-1, 1),
                             x_train_cat_number_only[c2].reshape(-1, 1), Y,c1 + " VS " + c2, "none")

def displayPlots(x_train, y_train):
    df_train = x_train.copy()
    df_train['Vote'] = y_train.copy().values
    describeAndPlot(df_train)

def chooseColumns(currCols,oldCols):
    modeSuffix = 'FillByModeInt'
    onehotSuffix = 'FillByMode_'
    meanSuffix = 'FillByMean'
    meadianSuffix = 'FillByMedian'
    nanSuffix = '_nan'
    # currList= list of original cols name, res= set of cols to continue with
    res = [colName for colName in currCols if onehotSuffix in colName]
    res += [colName for colName in currCols if (meadianSuffix in colName) & (meanSuffix not in colName)]
    currListMode = [colName for colName in oldCols if (colName + modeSuffix) in currCols]
    res += [(colName + modeSuffix) for colName in currListMode]
    currListMean = [colName for colName in oldCols if ((colName + meanSuffix) in currCols) & (colName not in currListMode)]
    res += [(colName + meanSuffix) for colName in currListMean if (colName + meadianSuffix) not in res]
    res = [colName for colName in res if nanSuffix not in colName]
    return res




"""
TODO: WORKFLOW
1) Fill in missing values in Category type columns by mode
2) Convert into One-hot and ObjectInt
3) Detect and remove outliers
4) Scale to Normal or MinMax
5) Fill in missing values by close samples
6) Fill in missing values in numeric columns by mean (mean label for train, mean of column for test/val
7) Leave only columns after analysis
8) Feature Selection by Filter method
8) Relief 
9) Tree
10)SFS
11)Accuracy on valdiation data
"""
def main():
    # read data from file
    df = pd.read_csv("./ElectionsData.csv")
    oldCols = df.columns


    df['IncomeMinusExpenses'] = df.Yearly_IncomeK - df.Yearly_ExpensesK # new column

    # seperate labels from data
    X = df.drop('Vote', axis=1)
    Y = pd.DataFrame(df['Vote'])

    # Split to train, valid, test
    np.random.seed(0)
    x_train, x_testVal, y_train, y_testVal = train_test_split(X, Y)
    x_val, x_test, y_val, y_test = train_test_split(x_testVal, y_testVal, train_size=0.6, test_size=0.4)


    # save before any changes
    x_train_final = x_train
    x_train_final['Vote'] = y_train.values
    x_val_final = x_val
    x_val_final['Vote'] = y_val.values
    x_test_final = x_test
    x_test_final['Vote'] = y_test.values
    # Save labels
    x_train_final.to_csv("./x_train_final.csv")
    x_val_final.to_csv("./x_val_final.csv")
    x_test_final.to_csv("./x_test_final.csv")
    y_train.to_csv("./y_train.csv")
    y_val.to_csv("./y_val.csv")
    y_test.to_csv("./y_test.csv")



    # Fill nan in category type TODO Step 1
    print('Entering stage 1: Fill in missing values in Category type columns by mode')
    colsCategory = x_train.select_dtypes(include=['object'])
    for col in colsCategory:
        x_train = fillNAByLabelMode(x_train,y_train,col)
        x_val = fillNATestValMode(x_val,col)
        x_test = fillNATestValMode(x_test,col)


    print('Entering stage 2: Convert into One-hot and ObjectInt')
    # Convert data to ONE-HOT & CATEGORY TODO Step 2
    x_train_cat, y_train_cat, x_val_cat, y_val_cat, x_test_cat, y_test_cat = \
        setTypesToCols(x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy(), x_test.copy(), y_test.copy())


    print('Entering stage 3: Detect and remove outliers')
    # Detect and remove outliers TODO Step 3
    outlierMap = {'Phone_minutes_10_years':(400,500000),'Avg_size_per_room':(12,None),
                  'Avg_monthly_income_all_years':(None,500000)}
    for col,boundaries in outlierMap.items():
        x_train_cat = changeOutlierToMean(x_train_cat,y_train_cat,col,'Yellows',boundaries[0],boundaries[1])


    # List of columns to normalize with scaleNormalSingleColumn. For others use MinMax scale
    colsToScaleNorm = ["Political_interest_Total_Score","Yearly_IncomeK","Avg_monthly_household_cost","Avg_size_per_room",
                       "Avg_monthly_expense_on_pets_or_plants","Avg_monthly_expense_when_under_age_21",
                       "AVG_lottary_expanses","Phone_minutes_10_years","Garden_sqr_meter_per_person_in_residancy_area",
                       "Avg_Satisfaction_with_previous_vote","Overall_happiness_score","Weighted_education_rank",
                       "Avg_monthly_income_all_years"]


    # Iterate over columns and scale them TODO Step 4
    print('Entering stage 4: Scale to Normal or MinMax')
    for colToScale in colsToScaleNorm: # scale by normal
        x_train_cat = stats.scaleNormalSingleColumn(x_train_cat,colToScale)
        x_val_cat = stats.scaleNormalSingleColumn(x_val_cat,colToScale)
        x_test_cat = stats.scaleNormalSingleColumn(x_test_cat,colToScale)

    colsToScaleMinMax = x_train_cat.select_dtypes(include=[np.number]).columns.difference(colsToScaleNorm) # rest of numeric cols
    for colToScale in colsToScaleMinMax: # scale by MINMAX
        x_train_cat = stats.scaleMinMaxSingleColumn(x_train_cat,colToScale)
        x_val_cat = stats.scaleMinMaxSingleColumn(x_val_cat,colToScale)
        x_test_cat = stats.scaleMinMaxSingleColumn(x_test_cat,colToScale)


    # List of relations between columns, according to Pearson and MI
    colToColRel = [["Avg_size_per_room", "Political_interest_Total_Score", "Yearly_IncomeK", "Avg_monthly_household_cost"],
                    ["AVG_lottary_expanses", "Avg_monthly_income_all_years", "Avg_monthly_expense_when_under_age_21", "Avg_Satisfaction_with_previous_vote", "Will_vote_only_large_partyFillByMode_Yes", "Will_vote_only_large_partyFillByMode_No", "Looking_at_poles_resultsFillByModeInt"],
                    ["Last_school_grades", "Will_vote_only_large_partyFillByMode_Maybe", "Most_Important_IssueFillByMode_Education", "Most_Important_IssueFillByMode_Military"],
                    ["Avg_monthly_expense_on_pets_or_plants", "MarriedFillByModeInt", "Garden_sqr_meter_per_person_in_residancy_area", "Phone_minutes_10_years"]]


    # Fill nan by relations TODO Step 5
    print('Entering stage 5: Fill in missing values by close samples')
    for relation in colToColRel:
        # x_train_cat.info()
        print('rel=',relation)
        x_train_cat.update(fillNanWithOtherColumns(x_train_cat,y_train_cat,relation))
        # x_train_cat.info()

    x_train_cat.to_csv("./afterRelations.csv")


    # Fill nan in numeric type TODO Step 6
    print('Entering stage 6: Fill in missing values in numeric columns by mean')
    colsFilledWithMode = x_train_cat.select_dtypes(include=[np.number]).columns
    suffix = 'FillByMode'
    colsFilledWithMode = [colName for colName in colsFilledWithMode if suffix in colName]
    colsToMedian = ['Num_of_kids_born_last_10_years','Number_of_valued_Kneset_members',
                    'Number_of_differnt_parties_voted_for']
    for col in colsToMedian:
        x_train_cat = fillNAByLabelMeanMedian(x_train_cat,y_train_cat,col,'Median')
        x_val_cat = fillNATestValMeanMedian(x_val_cat,col,'Median')
        x_test_cat = fillNATestValMeanMedian(x_test_cat,col,'Median')

    colsToMean = x_train_cat.select_dtypes(include=[np.number]).columns.difference(colsToMedian)
    colsToMean = colsToMean.difference(colsFilledWithMode)
    for col in colsToMean:
        x_train_cat = fillNAByLabelMeanMedian(x_train_cat,y_train_cat,col,'Mean')
        x_val_cat = fillNATestValMeanMedian(x_val_cat,col,'Mean')
        x_test_cat = fillNATestValMeanMedian(x_test_cat,col,'Mean')

    x_train_cat.to_csv("./x_train_cat.csv")
    x_val_cat.to_csv("./x_val_cat.csv")
    x_test_cat.to_csv("./x_test_cat.csv")


    ### Methdic Stop - start again from here###

    x_train_cat = pd.read_csv("./x_train_cat.csv",index_col=0)
    y_train_cat = pd.read_csv("./y_train.csv",index_col=0)


    # Remove features for the first time TODO Step 7
    print('Entering stage 7: Leave only columns after analysis')
    colsAfterWork = chooseColumns(x_train_cat.columns,oldCols)

    pd.DataFrame(colsAfterWork).to_csv("./listOfFeatsNames.csv")
    x_train_cat = x_train_cat[colsAfterWork]
    partiesLabels = y_train_cat['Vote'].unique()

    # pd.DataFrame(colsAfterWork).to_csv("./selected.csv")
    # print(len(colsAfterWork))
    # exit(4)

    # Relief Algorithm TODO Step 8
    print('Entering stage 8: Relief Algorithm')
    numOfSamplesForRelief = [5]
    for i in numOfSamplesForRelief:
        print('Starting relief with',i,'iterations')
        reliefRes = reliefFeatureSelection(x_train_cat, y_train_cat,numOfRowsToSample=i)
        pd.DataFrame(reliefRes).to_csv("./relief"+str(i)+".csv")

    ## tring feafures selected by RELIEF
    # featuresFromRelief = ["Most_Important_IssueFillByMode_Financial","OccupationFillByMode_Student_or_Unemployed",
    #                       "Main_transportationFillByMode_Motorcycle_or_truck","Financial_agenda_mattersFillByModeInt",
    #                       "Number_of_valued_Kneset_membersFillByMedian","AVG_lottary_expansesFillByMean",
    #                       "Voting_TimeFillByModeInt","Avg_monthly_income_all_yearsFillByMean","GenderFillByModeInt",
    #                       "Will_vote_only_large_partyFillByMode_No","Will_vote_only_large_partyFillByMode_Yes",
    #                       "Looking_at_poles_resultsFillByModeInt","Avg_monthly_household_costFillByMean",
    #                       "Avg_monthly_expense_on_pets_or_plantsFillByMean","Garden_sqr_meter_per_person_in_residancy_areaFillByMean",
    #                       "Political_interest_Total_ScoreFillByMean","Yearly_IncomeKFillByMean","Phone_minutes_10_yearsFillByMean",
    #                       "Avg_size_per_roomFillByMean","Weighted_education_rankFillByMean","Overall_happiness_scoreFillByMean"]
    #
    #
    # estimator = KNeighborsClassifier(n_neighbors=5)
    # print("Accuracy of Relief:",scoreForClassfier(clf=estimator,examples=x_train_cat[featuresFromRelief],classification=y_train_cat))
    #
    # exit(8)


    # Decision tree Algorithm TODO Step 9
    print('Entering stage 9: Decision tree Algorithm')
    treeRankedFeatures,confusionMatrix = embeddedDecisionTree(x_train_cat, y_train_cat)
    # make nice confusion matrix with labels
    confusionMatrix = pd.DataFrame(confusionMatrix,columns=partiesLabels,index=partiesLabels)
    print('The Confusion Matrix of tree is:\n',confusionMatrix)
    confusionMatrix.to_csv('./confusionMatrixTree.csv')
    print('Features ranked by Decision tree:\n',treeRankedFeatures)
    pd.DataFrame(treeRankedFeatures).to_csv("./treeRankedFeatures.csv")


    # SFS Algorithm TODO Step 10
    print('Entering stage 10: SFS Algorithm')
    # estimator = KNeighborsClassifier(n_neighbors=5)
    estimator = tree.DecisionTreeClassifier(criterion="entropy")
    # print('Accuracy from KNN:',scoreForClassfier(estimator,x_train_cat,y_train_cat))

    numOfFeaturesToSelect = 30
    sfsSelection = sfs(x_train_cat,y_train_cat,k=numOfFeaturesToSelect,
                       clf=estimator,score=scoreForClassfier)
    print(sfsSelection)
    sfsFeaturesNames = []
    for index in sfsSelection:
        sfsFeaturesNames.append(colsAfterWork[index])
    print('SFS has selected those features in this order:\n',sfsFeaturesNames)
    pd.DataFrame(sfsFeaturesNames).to_csv("./sfsFeaturesNameTree.csv")



    hybridFeatures = pd.read_csv("finalSelection.csv")
    hybridFeatures = list(hybridFeatures['Final_selection'])
    x_train_cat = x_train_cat[hybridFeatures]
    # Accuracy on validation-set TODO Step 11
    print('Entering stage 11: \tD-Day\nAccuracy on validation-set')
    # Load validation/test-set after preprocessing workflow
    x_val_cat = pd.read_csv("x_val_cat.csv",index_col=0)
    y_val_cat = pd.read_csv("y_val.csv",index_col=0)
    x_test_cat = pd.read_csv("x_test_cat.csv",index_col=0)
    y_test_cat = pd.read_csv("y_test.csv",index_col=0)

    x_val_cat = x_val_cat[hybridFeatures]
    x_test_cat = x_test_cat[hybridFeatures]


    # save after all changes and the final features selection
    x_train_final = x_train_cat
    x_train_final['Vote'] = y_train_cat.values

    x_val_final = x_val_cat
    x_val_final['Vote'] = y_val_cat.values

    x_test_final = x_test_cat
    x_test_final['Vote'] = y_test_cat.values

    # Save
    x_train_final.to_csv("./x_train_final.csv")
    x_val_final.to_csv("./x_val_final.csv")
    x_test_final.to_csv("./x_test_final.csv")

    # train knn and tree and test on validation
    estimator = KNeighborsClassifier(n_neighbors=5)
    estimator.fit(x_train_cat, y_train_cat)
    accuracy = accuracy_score(y_val_cat, estimator.predict(x_val_cat))
    print("The KNN classifier accuracy is:",accuracy)
    estimator = tree.DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train_cat, y_train_cat)
    accuracy = accuracy_score(y_val_cat, estimator.predict(x_val_cat))
    print("The Tree classifier accuracy is:", accuracy)



    # display plots
    # displayPlots(x_train, y_train)




    # x_train_cat_number_only = x_train_cat.select_dtypes(include=np.number)
    # 
    # x_train_cat_number_only['Vote'] = y_train_cat
    # x_train_cat_number_only = x_train_cat_number_only.dropna(axis=0, how='any')
    # 
    # stats.plotPCA(x_train_cat_number_only.drop(columns=['Vote']), x_train_cat_number_only['Vote'])
    #

    # calculate MI between each feature and label
    # for c in x_train_cat_number_only.columns:
    #     if c != 'Vote':
    #         print(c, "vs label", stats.mutualInformation(x_train_cat_number_only['Vote'],
    #                                                      MinMaxScaler().fit_transform(x_train_cat_number_only[c].reshape(-1,1))))

    # create correlation matrices and graphs between each pair in feature list
    # res = creatColVsColCorrelationMatrix(x_train_cat_number_only)
    # drawColVsColScatterPlot(x_train_cat_number_only,x_train_cat_number_only['Vote'])



if __name__ == '__main__':
    main()

