import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# read data into data frame

df = pd.read_csv("./ElectionsData.csv")

X = df.drop('Vote',axis=1).values
Y = pd.DataFrame(df['Vote'])

np.random.seed(0)
X_train, X_testVer, y_train, y_testVer = train_test_split(X,Y)

X_ver, X_test, y_ver, y_test = train_test_split(X_testVer,y_testVer,train_size=0.6,test_size=0.4)



