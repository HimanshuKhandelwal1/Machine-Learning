# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:01:43 2018

@author: HI369091
"""

#Importing all the important library of python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading dataset
dataset = pd.read_csv('Position_Salaries.csv')

#Separating the independent and dependent variable
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

#performing feature scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

#fitting SVR to the dataset
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X,y)

#predicting the value
y_pred = sc_y.inverse_transform(svr.predict(sc_x.transform(np.array([[6.5]]))))

#visualizing the SVR result
plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(svr.predict(X)), color = 'blue')
plt.title('SVR model')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()