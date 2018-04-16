import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def setTypesToCols(trainX:pd.DataFrame, trainY:pd.DataFrame,
                   validX: pd.DataFrame, validY: pd.DataFrame,
                   testX: pd.DataFrame, testY: pd.DataFrame):

    colToCategorial = trainX.keys()[trainX.dtypes.map(lambda x: x=='object')]

    colTocategoryOnehot = ["Most_Important_Issue", "Will_vote_only_large_party", "Main_transportation",
                           "Occupation"]

    for f in colToCategorial:
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

    return trainX, trainY, validX, validY, testX, testY


df = pd.read_csv("./ElectionsData.csv")
X = df.drop('Vote', axis=1).values
Y = pd.DataFrame(df['Vote'])

np.random.seed(0)
x_train, x_testVer, y_train, y_testVer = train_test_split(X, Y)

x_ver, x_test, y_ver, y_test = train_test_split(x_testVer, y_testVer, train_size=0.6, test_size=0.4)


x_train_cat, y_train_cat, x_ver_cat, y_ver_cat, x_test_cat, y_test_cat = \
    setTypesToCols(x_train.copy(), y_train.copy(), x_ver.copy(), y_ver.copy(), x_test.copy(), y_test.copy())

print(x_test_cat)