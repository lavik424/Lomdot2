
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from util import *
import stats
from featureSelection import *




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

    colOfMultiCategorial = ["Most_Important_Issue", "Will_vote_only_large_party", "Main_transportation",
                           "Occupation"]

    colOfOrderedCategorial = ["Age_group"]

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

"""
TODO: WORKFLOW
0) Fill in missing values in Category type columns by mode
1.1) Convert into One-hot and ObjectInt
1.2) Detect and remove outliers
2) Scale to Normal or MinMax
3) Fill in missing values by close samples
5) Fill in missing values in nueric columns by mean (mean label for train, mean of column for test/val
6) Feature Selection by Filter method
7) Relief 
8) Tree
9) SFS
10)Accuracy on valdiation data
"""
def main():
    # read data from file
    df = pd.read_csv("./ElectionsData.csv")
    oldCols = df.columns
    df.info()
    exit(3)


    df['IncomeMinusExpenses'] = df.Yearly_IncomeK - df.Yearly_ExpensesK # new column

    # seperate labels from data
    X = df.drop('Vote', axis=1)
    Y = pd.DataFrame(df['Vote'])

    # Split to train, valid, test
    np.random.seed(0)
    x_train, x_testVer, y_train, y_testVer = train_test_split(X, Y)
    x_val, x_test, y_val, y_test = train_test_split(x_testVer, y_testVer, train_size=0.6, test_size=0.4)

    print('Entering stage 1')
    # Convert data to ONE-HOT & CATEGORY TODO 1
    x_train_cat, y_train_cat, x_ver_cat, y_ver_cat, x_test_cat, y_test_cat = \
        setTypesToCols(x_train.copy(), y_train.copy(), x_val.copy(), y_val.copy(), x_test.copy(), y_test.copy())

    # Detect and remove outliers TODO 1.2
    outlierMap = {'Phone_minutes_10_years':(400,500000),'Avg_size_per_room':(12,None),
                  'Avg_monthly_income_all_years':(None,500000)}
    # x_temp = x_train_cat
    # x_temp['Vote'] = y_train_cat.values
    # for col,boundaries in outlierMap.items():
    #     x_temp['Vote'] = y_train_cat.values
    #     print(np.min(x_temp.loc[x_temp['Vote'] == 'Yellows',col]))
    #     x_temp = x_temp.drop('Vote', axis=1)
    #     x_temp = changeOutlierToMean(x_temp,y_train_cat,col,'Yellows',boundaries[0],boundaries[1])
    #     x_temp['Vote'] = y_train_cat.values
    #     print(np.min(x_temp.loc[x_temp['Vote'] == 'Yellows', col]))
    #     x_temp = x_temp.drop('Vote', axis=1)
    # exit(2)
    for col,boundaries in outlierMap.items():
        x_train_cat = changeOutlierToMean(x_train_cat,y_train_cat,col,'Yellows',boundaries[0],boundaries[1])

    # List of columns to normalize with scaleNormalSingleColumn. For others use MinMax scale

    colsToScaleNorm = ["Political_interest_Total_Score","Yearly_IncomeK","Avg_monthly_household_cost","Avg_size_per_room",
                       "Avg_monthly_expense_on_pets_or_plants","Avg_monthly_expense_when_under_age_21",
                       "AVG_lottary_expanses","Phone_minutes_10_years","Garden_sqr_meter_per_person_in_residancy_area",
                       "Avg_Satisfaction_with_previous_vote","Overall_happiness_score","Weighted_education_rank",
                       "Avg_monthly_income_all_years"]

    # Iterate over columns and scale them TODO 2
    print('Entering stage 2')
    for colToScale in colsToScaleNorm: # scale by normal
        x_train_cat = stats.scaleNormalSingleColumn(x_train_cat,colToScale)
        x_ver_cat = stats.scaleNormalSingleColumn(x_ver_cat,colToScale)
        x_test_cat = stats.scaleNormalSingleColumn(x_test_cat,colToScale)

    colsToScaleMinMax = x_train_cat.select_dtypes(include=[np.number]).columns.difference(colsToScaleNorm)
    for colToScale in colsToScaleMinMax: # scale by MINMAX
        x_train_cat = stats.scaleMinMaxSingleColumn(x_train_cat,colToScale)
        x_ver_cat = stats.scaleMinMaxSingleColumn(x_ver_cat,colToScale)
        x_test_cat = stats.scaleMinMaxSingleColumn(x_test_cat,colToScale)


    # List of relations between columns, according to Pearson and MI
    colToColRel = [["Avg_size_per_room", "Political_interest_Total_Score", "Yearly_IncomeK", "Avg_monthly_household_cost"],
                    ["AVG_lottary_expanses", "Avg_monthly_income_all_years", "Avg_monthly_expense_when_under_age_21", "Avg_Satisfaction_with_previous_vote", "Will_vote_only_large_party_Yes", "Will_vote_only_large_party_No", "Looking_at_poles_resultsInt"],
                    ["Last_school_grades", "Will_vote_only_large_party_Maybe", "Most_Important_Issue_Education", "Most_Important_Issue_Military"],
                    ["Avg_monthly_expense_on_pets_or_plants", "MarriedInt", "Garden_sqr_meter_per_person_in_residancy_area", "Phone_minutes_10_years"]]


    # Fill nan by relations TODO 3
    print('Entering stage 3')
    for relation in colToColRel:
        # x_train_cat.info()
        x_train_cat.update(fillNanWithOtherColumns(x_train_cat,y_train_cat,relation))
        # x_train_cat.info()

    x_train_cat.to_csv("./afterRelations.csv")


    # Fill nan in category type TODO 4
    print('Entering stage 4')
    colsCategory = x_train_cat.select_dtypes(include=['category'])
    for col in colsCategory:
        x_train_cat = fillNAByLabelMode(x_train_cat,y_train_cat,col)
        x_ver_cat = fillNATestValMode(x_ver_cat,col)
        x_test_cat = fillNATestValMode(x_test_cat,col)


    # Fill nan in numeric type TODO 5
    print('Entering stage 5')
    colsToMean = x_train_cat.select_dtypes(include=[np.number])
    for col in colsToMean:
        x_train_cat = fillNAByLabelMeanMedian(x_train_cat,y_train_cat,col,'Mean')
        x_ver_cat = fillNATestValMeanMedian(x_ver_cat,col,'Mean')
        x_test_cat = fillNATestValMeanMedian(x_test_cat,col,'Mean')

    x_train_cat.to_csv("./afterCatNum.csv")
    exit(3)


    # Remove features for the first time TODO 6
    print('Entering stage 6')
    colsToDrop = oldCols + []
    x_train_cat_filtered = x_train_cat.drop(colsToDrop)

    # From now use numeric cols only !!!
    x_train_cat_filtered = x_train_cat_filtered.select_dtypes(include=[np.number])








    # display plots
    # displayPlots(x_train, y_train)



    ## TEST NEARESTHITMISS
    # nearesthit = findNearestHitMiss(x_train,y_train,1,'h')
    # print(x_train.iloc[[1]])
    # print('nearestHit:\n',x_train.loc[[nearesthit]])
    # nearestmiss = findNearestHitMiss(x_train, y_train, 1, 'm')
    # print('nearestMiss:\n', x_train.loc[[nearestmiss]])
    
    
    ## TEST RELIEF ALGORITM
    # print(reliefFeatureSelection(x_train,y_train))


    # # Replace nan with mean
    # colToInt = x_train.select_dtypes(include=[np.number]).columns
    # for col in colToInt:
    #     x_train = fillNAByLabelMeanMedian(x_train.copy(),y_train,col,'Mean')
    #
    # # LEAVE only numeric columns w/o nan
    # x_train = x_train.select_dtypes(include=[np.number])
    # x_train = x_train.dropna(axis=1, how='any')
    #
    # # TEST EMBBDED FEATURE SELECTION BY DECISION TREE
    # embbdedDecisionTree(x_train,y_train) # not working needs data w/o nan


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

