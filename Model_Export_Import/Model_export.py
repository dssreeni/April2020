import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn.externals import joblib #For exporting and importing

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("E:/Data Science/Data")

titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#data preparation
titanic_train1 = pd.get_dummies(titanic_train, columns=['Pclass', 'Sex', 'Embarked'])
titanic_train1.shape
titanic_train1.info()
titanic_train1.head(6)

#feature engineering 
X_train = titanic_train1.drop(['PassengerId','Cabin','Ticket', 'Name','Survived'], 1)
y_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier(criterion = 'entropy')

dt_grid = {'criterion':['gini','entropy'], 'max_depth':list(range(3,15)), 'min_samples_split':[2,3,6,7,8]}
param_grid = model_selection.GridSearchCV(dt, dt_grid, cv=10) #Evolution of tee
param_grid.fit(X_train, y_train) #Building the tree
print(param_grid.best_score_) #Best score
print(param_grid.best_params_)
print(param_grid.score(X_train, y_train)) #train score  #Evolution of tree

#use cross validation to estimate performance of model. 
#==============================================================================
# cv_scores = model_selection. (dt, X_train, y_train, cv=5, verbose=3)
# cv_scores.mean()
#==============================================================================

#build final model on entire train data which is us for prediction
#dt.fit(X_train,y_train)

# natively deploy decision tree model(pickle format)
os.getcwd()
joblib.dump(param_grid, "TitanicVer2.1_C1.pkl")
