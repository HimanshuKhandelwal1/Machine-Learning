# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:00:16 2018

@author: HI369091
"""

#importing important libraries of python.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#Reading Dataset using Pandas library.
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.columns
dataset.describe(include= 'all')

# converting string value into the numerical in the dataset.
sex_mapping = {'Female':0,'Male':1}
dataset['Gender'] = dataset['Gender'].map(sex_mapping)

location_mapping = {'France':0,'Germany':1,'Spain':2}
dataset['Geography'] = dataset['Geography'].map(location_mapping)

#Separating independent and dependent variable
X = dataset.iloc[: ,3:13].values
y = dataset.iloc[: ,13:14].values

#Applying categorial Features
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1: ]

#creating training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20, random_state = 0)

# performing feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#K-Nearest Neighbors Regression
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)

#Calculating the accuracy score using accuracy_score metrics in sklearn library
knn_acc = round(accuracy_score(y_pred_knn,y_test)*100,2)