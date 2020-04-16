import os
import pandas as pd
from sklearn.externals import joblib

#changes working directory
os.chdir("E:/Data Science/Data")

#predict the outcome using decision tree
titanic_test = pd.read_csv("titanic_test.csv")
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()

titanic_test1 = pd.get_dummies(titanic_test, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)

#No prediction logic here

#Use load method to load Pickle file
serv2 = joblib.load("TitanicVer2.pkl")
titanic_test['Survived'] = serv2.predict(X_test)
titanic_test.to_csv("submissionUsingJobLib2.csv", columns=['PassengerId','Survived'], index=False)
