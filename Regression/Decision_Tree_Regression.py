# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:01:45 2018

@author: HI369091
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Separating the independent and dependent variable
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

#Fitting Decision Tree to the dataset
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(criterion = 'mse',random_state=0)
tree.fit(X,y)

#Predicting the value
y_pred = tree.predict(6.5)

#visualizing the Decision Tree regresssion in higher resolution
X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,tree.predict(X_grid), color= 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Salary')
plt.ylabel('Year of Experience')
plt.show()
