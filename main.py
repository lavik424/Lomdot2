import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from util import describeAndPlot




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
                   validX: pd.DataFrame, validY: pd.DataFrame,
                   testX: pd.DataFrame, testY: pd.DataFrame):

    colTocategoryOnehot = ["Most_Important_Issue", "Will_vote_only_large_party", "Main_transportation",
                           "Occupation"]

    colToCategoryOrdered = ["Age_group"]

    colToCategorialBinary = [c for c in trainX.keys()[trainX.dtypes.map(lambda x: x=='object')]
                       if c not in colTocategoryOnehot and c not in colToCategoryOrdered]

    ### translate ordered categorial
    f = colToCategorialBinary[0]
    # categorize train set
    trainX[f] = trainX[f].astype("category")
    trainX[f + "Int"] = trainX[f].cat.rename_categories(
        {'Below_30': 0, '30-45': 1, '45_and_up': 2})
    trainX.loc[trainX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

    # Let's creat a crosstabcross-tabulation to look at this transformation
    pd.crosstab(trainX[f+"Int"], trainX[f], rownames=[f+"Int"], colnames=[f])

    # categorize valid set
    validX[f] = validX[f].astype("category")
    if validX[f].cat.categories.isin(validX[f].cat.categories).all():
        validX[f] = validX[f].cat.rename_categories(validX[f].cat.categories)
    else:
        print("\n\nTrain and Valid don't share the same set of categories in feature '", f, "'")
    validX[f + "Int"] = validX[f].cat.rename_categories(
        {'Below_30': 0, '30-45': '1', '45_and_up': '2'})
    validX.loc[validX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

    # categorize test set
    testX[f] = testX[f].astype("category")
    if testX[f].cat.categories.isin(testX[f].cat.categories).all():
        testX[f] = testX[f].cat.rename_categories(testX[f].cat.categories)
    else:
        print("\n\nTrain and Test don't share the same set of categories in feature '", f, "'")
    testX[f + "Int"] = testX[f].cat.rename_categories(
        {'Below_30': 0, '30-45': '1', '45_and_up': '2'})
    testX.loc[testX[f].isnull(), f + "Int"] = np.nan  # fix NaN conversion

    ### translate binary categorial
    for f in colToCategorialBinary:
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
    for f in colTocategoryOnehot:
        # categorize train set
        trainX[f] = trainX[f].astype("category")
    trainX = pd.concat([pd.get_dummies(trainX, columns=colTocategoryOnehot, dummy_na=True),
                        trainX[colTocategoryOnehot]], axis=1)

    testX = pd.concat([fix_multivar_columns_for_test(testX,trainX,colTocategoryOnehot),
                       testX[colTocategoryOnehot]], axis=1)

    trainX = pd.concat([fix_multivar_columns_for_test(trainX,trainX,colTocategoryOnehot),
                        trainX[colTocategoryOnehot]], axis=1)

    return trainX, trainY, validX, validY, testX, testY




def main():
    df = pd.read_csv("./ElectionsData.csv")
    X = df.drop('Vote', axis=1)
    Y = pd.DataFrame(df['Vote'])


    np.random.seed(0)
    x_train, x_testVer, y_train, y_testVer = train_test_split(X, Y)


    x_ver, x_test, y_ver, y_test = train_test_split(x_testVer, y_testVer, train_size=0.6, test_size=0.4)



    x_train_cat, y_train_cat, x_ver_cat, y_ver_cat, x_test_cat, y_test_cat = \
        setTypesToCols(x_train.copy(), y_train.copy(), x_ver.copy(), y_ver.copy(), x_test.copy(), y_test.copy())



    df_train = x_train.copy()
    df_train['Vote'] = y_train.copy().values
    describeAndPlot(df_train)

    # print (x_train_cat)
    # print(x_train_cat.info())
    print(pd.get_dummies(x_train_cat).columns)
    # print(x_train_cat["MarriedInt"])


if __name__ == '__main__':
    main()

