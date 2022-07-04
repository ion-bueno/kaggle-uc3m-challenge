import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
from src.utils import seed


######################
# Missing values
######################

def fill_na_pmm(df_format=False):
    data = pd.read_csv('preprocessed/data_pmm.csv').iloc[:,2:]
    if df_format:
        return data
    return data.to_numpy()


def fill_na_median(X):
    for col in X:
        X[col] = X[col].fillna(X[col].median())
    return X


def fill_na_lr(X):
    while(X.isna().sum().sum() != 0):
        for col in X:
            df = X
            lr = LinearRegression()
            testdf = df[df[col].isna()==True]
            traindf = df[df[col].isna()==False]
            y = traindf[col]
            traindf = traindf.drop(col,axis=1)

            for c in traindf:
                traindf[c] = traindf[c].fillna(traindf[c].median())
                if c != col:
                    testdf[c] = testdf[c].fillna(traindf[c].median())

            lr.fit(traindf,y)

            testdf = testdf.drop(col,axis=1)

            pred = lr.predict(testdf)
            testdf[col]= pred

            for i in list(testdf.index):
                X.loc[i, col] = testdf.loc[i, col]
    return X



#########################
# Features selection
#########################


def remove_corr_features(X, corr=0.8):
    if type(X).__module__ == np.__name__:
        X = pd.DataFrame(X)
    correlated_features = []
    correlation_matrix = X.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > corr:
                colname = correlation_matrix.columns[i]
                correlated_features.append(colname)
    correlated_features
    return X.drop(columns=correlated_features)


def select_features(X):
    # dt
    #new_cols = ['feat_2','feat_4','feat_5','feat_7','feat_10','feat_12','feat_15','feat_16','feat_24','feat_25','feat_27','feat_28','feat_30','feat_31','feat_32']
    # rf
    #new_cols = ['feat_2','feat_4','feat_5','feat_6','feat_7','feat_10','feat_11','feat_12','feat_16','feat_17','feat_20','feat_22','feat_24','feat_25','feat_27','feat_28','feat_30','feat_31','feat_32','feat_34']
    # et
    new_cols = ['feat_2','feat_4','feat_5','feat_6','feat_7','feat_10','feat_11','feat_12','feat_16','feat_17','feat_18','feat_20','feat_22','feat_24','feat_25','feat_27','feat_28','feat_30','feat_31','feat_32','feat_34']
    return X[new_cols].to_numpy()

