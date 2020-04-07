#Data Frames1
#Python is case sensitive
import pandas as pd
titanic_train = pd.read_csv("E:/Data Science/Data/titanic_train.csv")
print(type(titanic_train))

#explore the dataframe
titanic_train.shape #No of rows and Column
titanic_train.info() #Data Type and nullable/non-nullable
titanic_train.describe() #Gives statistical information

#access column/columns of a dataframe
titanic_train['Sex']
titanic_train['Fare']
titanic_train.Sex
titanic_train.Fare
temp = titanic_train[['Survived','Fare', 'Embarked']]
print(type(temp))

#access rows of a data frame
titanic_train.iloc[855:865] #ith record

titanic_train[10:20] 
titanic_train.iloc[10:20]

titanic_train[885:891]
titanic_train.iloc[885:891]

#Get me top n records
titanic_train.head(10)
#Get me bottom n records
titanic_train.tail(10)

#access both rows and columns of a dataframe
titanic_train.iloc[10:11]

#If you wanted to access by column name then use .loc
titanic_train.loc[10:20,('Name', 'Age')]

#conditional access of dataframe
titanic_train.loc[titanic_train.Sex == 'female', 'Sex']
titanic_train.loc[titanic_train.Sex == 'female', 'Fare']

#grouping data in data frames
titanic_train.groupby(['Embarked']).size()
titanic_train.groupby(['Pclass', 'Sex']).size()
titanic_train.groupby(['Pclass', 'Embarked']).mean()

titanic_train.groupby(['Pclass', 'Embarked', 'Sex']).mean()['Fare']

