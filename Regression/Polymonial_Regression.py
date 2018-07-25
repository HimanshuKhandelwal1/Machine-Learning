# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:01:43 2018

@author: HI369091
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading Dataset using Pandas library.
dataset = pd.read_csv('Position_Salaries.csv')

#Separating independent and dependent variable
X = dataset.iloc[: ,1:2].values
y = dataset.iloc[: ,2:3].values

#Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X, y)

#Fitting polinomial Regression to the dataset
#it will transform X in to independent variable of power == degree
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
linear_reg = LinearRegression()
linear_reg.fit(X_poly,y)

#visualizing the linear Regression result
plt.scatter(X,y,color='red')
plt.plot(X,linear_regression.predict(X),color='blue')
plt.show()

#visualizing the polynomial Regression result
plt.scatter(X,y,color='red')
plt.plot(X,linear_reg.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#Prediction result in linear regression
linear_regression.predict(6.5)

#Prediction result in polynomial linear regression
linear_reg.predict(poly_reg.fit_transform(6.5))