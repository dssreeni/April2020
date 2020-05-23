# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

dft = df.tail(7)  #Bottom 5 Records
#==============================================================================
#      sepal_len  sepal_wid  petal_len  petal_wid           class
# 145        6.7        3.0        5.2        2.3  Iris-virginica
# 146        6.3        2.5        5.0        1.9  Iris-virginica
# 147        6.5        3.0        5.2        2.0  Iris-virginica
# 148        6.2        3.4        5.4        2.3  Iris-virginica
# 149        5.9        3.0        5.1        1.8  Iris-virginica
#==============================================================================

X = dft.ix[:,0:4].values #All rows and 0-4 columns which are nothing but 4 i/p features
y = dft.ix[:,4].values #All rows and 5th column which is nothing but o/p column(Class)
X_std = StandardScaler().fit_transform(X)
X.shape
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

#Alternate way to print CoVariance matrix
#print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

#Next, we perform an eigendecomposition on the covariance matrix
cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)




