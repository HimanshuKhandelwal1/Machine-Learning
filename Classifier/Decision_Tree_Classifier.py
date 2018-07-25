# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:58:12 2018

@author: HI369091
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
dataset.columns
dataset.describe(include= 'all')

# converting string value into the numerical.
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

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(X_train,y_train)
y_pred_tree = tree.predict(X_test)

#Calculating accuracy score
from sklearn.metrics import accuracy_score
tree_acc = round(accuracy_score(y_pred_tree, y_test)*100,2)

