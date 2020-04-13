# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""
#DecissionTree and Predict methods are very important in this example. This is the real starting/building of ML
#Here we will be playing with more columns. However DecisionTreeClassifier algorithm works only on numeric/continuous data/columns
#Henceforth we need to convert  catogerical columns to dummy columns
#This technique is called one-hot encoding

import pandas as pd
from sklearn import tree
import io
import pydot #if we need to use any external .exe files.... Here we are using dot.exe
import os
from sklearn import model_selection
os.chdir("E:/Data Science/Data/")

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

titanic_train = pd.read_csv("E:/Data Science/Data/titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#Convert categoric to One hot encoding using get_dummies
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.describe

#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'],1) 
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier()
#Build the decision tree model
param_grid = {'max_depth':[8, 10, 15], 'min_samples_split':[2, 4, 6], 'criterion':['gini', 'entropy']}

print(type(param_grid))
dt_grid = model_selection.GridSearchCV(dt, param_grid, cv=10, n_jobs=8)
#print(type(dt_grid))

#.fit builds the model.
#dt.fit(X_train,y_train)
dt_grid.fit(X_train,y_train) #Builds a decision tree classifier from the training set (X, y).
type(dt)

dt_grid.cv_results_
dt_grid.best_params_
dt_grid.best_score_ 

#==============================================================================
# #visualize the decission tree
# dot_data = io.StringIO()  
# 
# tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("DTwithParameters.pdf")
# 
#==============================================================================
#predict the outcome using decission tree
titanic_test = pd.read_csv("E:\\Data Science\\Data\\titanic_test.csv")
#titanic_test.info() #Found that one row has Fare = null in test data. Instead of dropping this column, let's take the mean of it.
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

#Now apply same get_dummies and drop columns on test data as well like above we did for train data
titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
#Apply the model on Furture/test data
titanic_test['Survived'] = dt_grid.predict(X_test)
titanic_test.to_csv("Submission_Grid.csv", columns=['PassengerId', 'Survived'], index=False)
