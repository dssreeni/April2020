#Objective based - Logistic Regression model(LR Model). It's for calssification
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model 
from sklearn import model_selection

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/Data Science/Data")
titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
X_train.info()
y_train = titanic_train['Survived']

#Logistic Regression is available in sklearn-->liner_model
lr_estimator = linear_model.LogisticRegression(random_state=2017)

#Parameters: C(Inverse of regularization strength) - Control of overfit parameter/Regulizer(regularization strength), How much of L1/L2
#Penalty is nothing but L1 or L2 (Regulizers)
#max_iter: How many iteration to try... Being it uses Guess & Refine approach internally
lr_grid = {'C':list(np.arange(0.1,1.0,0.19)), 'penalty':['l1', 'l2'], 'max_iter':list(range(20, 51, 10))}
#Remeber that we don't know which paramaeters for Controlling overfitting is good. We have keep try with different values,
#and check the diff between train score and CV score. Once they are close, then we can say that that's the best parameter.
#That means, you have to always compare train score and CV Score to come to optimal fitting
lr_grid_estimator = model_selection.GridSearchCV(lr_estimator, lr_grid, cv=10, n_jobs=1, verbose = 3)
lr_grid_estimator.fit(X_train, y_train)
lr_grid_estimator.grid_scores_
final_model = lr_grid_estimator.best_estimator_
lr_grid_estimator.best_score_
lr_grid_estimator.best_params_

final_model.coef_ 
final_model.intercept_

#Final Prections Preparation
titanic_test = pd.read_csv("titanic_test.csv")

#Note that you have to do the same work on test as well
titanic_test.shape
titanic_test.info()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
titanic_test1.shape
titanic_test1.info()
titanic_test1.head(6)
titanic_test1.describe()

X_test = titanic_test1.drop(['PassengerId', 'Age','Cabin','Ticket', 'Name'], 1)
X_test.info()

#create an instance of Imputer class with required arguments
mean_imputer = preprocessing.Imputer()  #Default value for Imputer is mean
mean_imputer.fit(X_test[['Fare']])
X_test[['Fare']] = mean_imputer.transform(X_test[['Fare']])
titanic_test['Survived'] = final_model.predict(X_test)
#titanic_test['Survived'] = lr_grid_estimator.predict(X_test)

titanic_test.to_csv('submission_lr_attempt2.csv', columns=['PassengerId','Survived'],index=False)
