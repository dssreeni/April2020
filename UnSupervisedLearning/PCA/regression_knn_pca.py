# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""

import os
import pandas as pd
#For K Nearest Neighbors
from sklearn import neighbors
from sklearn import feature_selection
from sklearn import preprocessing, ensemble
from sklearn import model_selection, metrics
import numpy as np
#sns.set_style('whitegrid') #Graphical representaton in a grid form
import math

#Visulaization package
import matplotlib.pyplot as plt
import seaborn as sns


def get_continuous_columns(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_columns(df):
    return df.select_dtypes(exclude=['number']).columns

def transform_cat_to_cont(df, features, mappings):
    for feature in features:
        null_idx = df[feature].isnull()
        df.loc[null_idx, feature] = None 
        df[feature] = df[feature].map(mappings)

def transform_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def get_missing_features(df):
    counts = df.isnull().sum()    
    return pd.DataFrame(data = {'features':df.columns,'count':counts,'percentage':counts/df.shape[0]}, index=None)

def get_missing_features1(df) :
    total_missing = df.isnull().sum()
    to_delete = total_missing[total_missing > 0]
    return list(to_delete.index)

def filter_features(df, features):
    df.drop(features, axis=1, inplace=True)

#Imputation means filling missing data
def get_imputers(df, features):
    all_cont_features = get_continuous_columns(df)
    cont_features = []
    cat_features = []
    for feature in features:
        if feature in all_cont_features:
            cont_features.append(feature)
        else:
            cat_features.append(feature)
    mean_imputer = preprocessing.Imputer() #Mean vlaue
    mean_imputer.fit(df[cont_features])
    mode_imputer = preprocessing.Imputer(strategy="most_frequent") #Mode
    mode_imputer.fit(df[cat_features])
    return mean_imputer, mode_imputer

def impute_missing_data(df, imputers):
    cont_features = get_continuous_columns(df)
    cat_features = get_categorical_columns(df)
    df[cont_features] = imputers[0].transform(df[cont_features])
    df[cat_features] = imputers[1].transform(df[cat_features])

def get_heat_map_corr(df):
    corr = df.select_dtypes(include = ['number']).corr()
    sns.heatmap(corr, square=False) #to get correlation among all dimensions
    plt.xticks(rotation=70)
    plt.yticks(rotation=70)
    return corr
#######
    
def get_target_corr(corr, target):
    return corr[target].sort_values(axis=0,ascending=False)

def one_hot_encode(df):
   features = get_categorical_columns(df)
   return pd.get_dummies(df, columns=features)

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_log_error(y_orig,y_pred) )
  
def fit_model(estimator, grid, X_train, y_train):
   grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring = metrics.make_scorer(rmse), cv=10, n_jobs=1)
   
   grid_estimator.fit(X_train, y_train)
   print(grid_estimator.cv_results_)
   #print(grid_estimator.best_params_)
   print(grid_estimator.best_score_)
   print(grid_estimator.score(X_train, y_train))
   return grid_estimator.best_estimator_

def feature_importances(estimator):
    return estimator.feature_importances_

def predict(estimator, X_test):
    return estimator.predict(X_test)

#to cover the mismatch of levels between train and test data
def merge(df1, df2):
    return pd.concat([df1, df2])

def split(df, ind):
    return (df[0:ind], df[ind:])

def viz_cont_cont(df, features, target):
    for feature in features:
        sns.jointplot(x = feature, y = target, data = df)
        
def viz_cat_cont_density(df, features, target):
    for feature in features:
        sns.FacetGrid(df, row=feature,size=8).map(sns.kdeplot, target).add_legend()
        plt.xticks(rotation=45)
        
        
def viz_cat_cont_box(df, features, target):
    for feature in features:
        sns.boxplot(x = feature, y = target,  data = df)
        plt.xticks(rotation=30)        

def viz_cont(df, features):
    for feature in features:
        sns.distplot(df[feature],kde=False)

def feature_selection_from_model(estimator, feature_data, target):
    estimator.fit(feature_data, target)
    
    features = pd.DataFrame({'feature':feature_data.columns, 'importance':estimator.feature_importances_})
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    features.plot(kind='barh', figsize=(20, 20))

    fs_model = feature_selection.SelectFromModel(estimator, threshold="mean", prefit=True)
    return features, fs_model.transform(feature_data)

def get_scale_model(df) :
    scaler = preprocessing.StandardScaler()
    scaler.fit(df)
    return scaler

#####
os.chdir("E:/Data Science/Data")

house_train = pd.read_csv("house_train.csv")
house_train.shape
house_train.info()

house_test = pd.read_csv("house_test.csv")
house_test.shape
house_test.info()
house_test['SalePrice'] = 0

house_data = merge(house_train, house_test)
house_data.shape
house_data.info()

print(get_continuous_columns(house_data))
print(get_categorical_columns(house_data))

#convert numerical columns to categorical type              
features = ['MSSubClass']
transform_cont_to_cat(house_data, features)

#map string categoical values to numbers
ordinal_features = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "PoolQC", "FireplaceQu", "KitchenQual", "HeatingQC"]
quality_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
transform_cat_to_cont(house_data, ordinal_features, quality_dict)

#filter missing data columns
missing_features = get_missing_features1(house_data)
filter_features(house_data, missing_features)
house_data.shape
house_data.info()

#explore relation among all continuous features
corr = get_heat_map_corr(house_train)
get_target_corr(corr, 'SalePrice')

house_train, house_test = split(house_data, house_train.shape[0])
house_train.shape
house_test.shape

features = ['SalePrice']
viz_cont(house_train, features)

#explore relationship of neighborhood to saleprice
#Categoric to Continuous relation
target = 'SalePrice'
features = ['BldgType']
viz_cat_cont_box(house_train, features, target)

#explore relationship of livarea and totalbsmt to saleprice
features = ['GrLivArea']
target = 'SalePrice'
viz_cont_cont(house_train, features, target)

features_to_filter = ['Id']
                               
#do one-hot-encoding for all the categorical features
house_train1 = one_hot_encode(house_train)
house_train1.shape
house_train1.info()

#filter_features(house_train1, ['SalePrice','log_sale_price'])
filter_features(house_train1, ['SalePrice'])
X_train = house_train1
y_train = house_train['SalePrice']

X_train.shape
#Step 1 
rf_estimator = ensemble.RandomForestClassifier(n_estimators=50, verbose = 3)
feature_imp_df, X_train1 = feature_selection_from_model(rf_estimator, X_train, y_train)
#X_train1.shape

from sklearn import decomposition
#PCA
#Let's apply PCA with out standard scalar
pca = decomposition.PCA(10)
pca.fit(house_train1)
pca.explained_variance_
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()  
house_train3 = pca.transform(house_train1)

#Let's apply PCA with standard scalar
scaler = preprocessing.StandardScaler()
scaler.fit(house_train1)
house_train2 = scaler.transform(house_train1)
pca = decomposition.PCA(10)
pca.fit(house_train2)
pca.explained_variance_
pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()  
house_train3 = pca.transform(house_train2)

x_train_Scaled = house_train3
house_train['log_sale_price'] = np.log(house_train['SalePrice'])
y_train = house_train['log_sale_price']

#==============================================================================

knn_estimator = neighbors.KNeighborsRegressor()
knn_grid = {'n_neighbors':[5,10,15]}
model = fit_model(knn_estimator, knn_grid, x_train_Scaled, y_train) #on PCAd data

#Best score: 0.182414942974
#.score: 0.17392101913392907
