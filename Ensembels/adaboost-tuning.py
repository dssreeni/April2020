import os
import pandas as pd
from sklearn import ensemble
from sklearn import tree
from sklearn import model_selection
import pydot
import io

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/Data Science/Data")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#Note that we take entire data into consideration in boosting. That's why we have to cut the tree depth to control the overfitting. 
#In this case we are giving max_depth=3
dt_estimator = tree.DecisionTreeClassifier(max_depth=3)
ada_tree_estimator1 = ensemble.AdaBoostClassifier(dt_estimator, 5)

#Parameter tuning
#n_estimators(no. of trees to grow), learning_rate(Learning rate shrinks the contribution of each classifier by learning_rate.)
#There is a trade-off between learning_rate and n_estimators.
#Pass the learning_rate less than default which is 1.
ada_grid = {'n_estimators':[5, 8, 10, 12],'learning_rate':[0.1, 0.5, 0.9]}
ada_grid_estimator = model_selection.GridSearchCV(ada_tree_estimator1,ada_grid, cv=10, n_jobs=1)
ada_grid_estimator.fit(X_train, y_train)
ada_grid_estimator.cv_results_
ada_grid_estimator.best_score_
ada_grid_estimator.best_params_

best_est = ada_grid_estimator.best_estimator_

#extracting all the trees build by adaboost algorithm
n_tree = 0
for est in best_est.estimators_: 
    dot_data = io.StringIO()
    #tmp = est.tree_
    tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())#[0] 
    graph.write_pdf("adatree" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1