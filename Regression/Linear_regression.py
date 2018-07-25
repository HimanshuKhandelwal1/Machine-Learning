# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:01:43 2018

@author: HI369091
"""

#importing important libraries of python.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Reading Dataset using Pandas library.
dataset = pd.read_csv('Salary_Data.csv')

#Separating independent and dependent variable
X = dataset.iloc[: ,0:1].values
y = dataset.iloc[: ,1:2].values

#creating training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train,y_train)

#Predicting the test set result.
y_pred = linear_regression.predict(X_test)

# visualizing the training set 
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,linear_regression.predict(X_train),color = 'blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the test set
plt.scatter(X_test,y_test,color='Red')
plt.plot(X_train, linear_regression.predict(X_train),color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
