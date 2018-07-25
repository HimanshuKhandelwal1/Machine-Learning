# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:03:17 2018

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

#Fitting RandomForest to the dataset
from sklearn.ensemble import RandomForestRegressor
Forest_reg = RandomForestRegressor(n_estimators = 300, criterion = 'mse',random_state = 0)
Forest_reg.fit(X,y)

#predict the new results
y_pred = Forest_reg.predict(6.5)

#visualizing the result
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,Forest_reg.predict(X_grid),color='blue')
plt.show()