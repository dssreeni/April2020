# -*- coding: utf-8 -*-
"""
Model tuning
@author: Sreenivas.J
"""

import pandas as pd
from sklearn import tree
from sklearn import model_selection
import io
import pydot
import os
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/Data Science/Data/")

titanic_train = pd.read_csv("titanic_train.csv")
print(type(titanic_train))

#EDA
titanic_train.shape
titanic_train.info()

#Apply one hot encoding
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#Model 1A
dt1 = tree.DecisionTreeClassifier()
dt1.fit(X_train,y_train)
#Apply K-fold technique and find out the Cross Validation(CV) score.
cv_scores1 = model_selection.cross_val_score(dt1, X_train, y_train, cv=9)
print(cv_scores1) #Return type is a [List] of score
print(cv_scores1.mean()) #Find out the mean of CV scores

#Model 1B
print(dt1.score(X_train,y_train))

#Section 2A
#tune model manually by passing differnt values for decision tree arguments
dt2 = tree.DecisionTreeClassifier(max_depth=4) #Here we passed max-depth as argument to the tree
dt2.fit(X_train,y_train)
cv_scores2 = model_selection.cross_val_score(dt2, X_train, y_train, cv=9)
print(cv_scores2) #Return type is a [List] of scores
print(cv_scores2.mean())

#Section 2B
print(dt2.score(X_train,y_train))

#Automate model tuning process. Use Grid search method
dt3 = tree.DecisionTreeClassifier()
param_grid = {'max_depth':[5,8,10], 'min_samples_split':[2,4,5], 'criterion':['gini','entropy']} 
#print(param_grid)
#max_depth means: Max deapth of the tree to child nodes
#min_samples_split means: If you notice the tree nodes, there is some thing called sample in each node. This is what it is referring to min sample split
dt3_grid = model_selection.GridSearchCV(dt3, param_grid, cv=9, n_jobs=5)
dt3_grid.fit(X_train, y_train)

print(dt3_grid.cv_results_) #You may get DeprecationWarning
#print(dt3_grid.cv_results_) #New version
final_model = dt3_grid.best_estimator_ #This is the estimator of max_deapth and min_sample_split combination
print(dt3_grid.best_score_)
#.score gives the score on full train data
print(dt3_grid.score(X_train, y_train))

dot_data = io.StringIO() 
tree.export_graphviz(final_model, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Tuned_DT.pdf")
