#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:25:01 2023

@author: blue
"""

#%% Script outline
# Data cleanup and normalization - Aaron
# Train/dev/test split - Aaron
# Model building
# * KNN - Mansi
# * Logistic Regression - Natalie
# * SVM - Grace
# * Neural Network - Niru
# Evaluation - Aaron

#%%
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.svm import SVC 
from sklearn import svm #importing svm model
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV


#%% Create Directories
# Create paths to each directory
file_names = ['diabetes.csv']
p_data = file_names[0]

# Create any missing directories
for i in file_names:
    if not os.path.exists(i):
        print('Please only run script in same directory as diabetes.csv file.')
        exit()


#%% Data cleanup and normalization
# Read in data
pima = pd.read_csv('diabetes.csv')

# Convert predictors to 1's and -1's
pima['Outcome'][pima['Outcome'] == 0] = -1


# Store the training values
X = pima.iloc[:,:-1].values
Y = pima.iloc[:,-1].values

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to the data and transform it
X_norm = scaler.fit_transform(X)


#%% Train/dev/test split

# First seperate the training set from the dev/test (temp) sets
train_X, temp_X, train_y, temp_y = train_test_split(X_norm, Y, test_size = 0.30, shuffle = True)
# Then split that temporary set in half to create dev/test sets
dev_X, test_X, dev_Y, test_y = train_test_split(temp_X, temp_y, test_size = 0.50, shuffle = False)



#%% Model building




#%% KNN





#%% Logistic Regresson




#%% SVM
#Creating SVM classifier with linear kernel
classifier = svm.SVC(kernel='rbf') 

#Fitting the model with the training sets
classifier.fit(train_X,train_y)

#K-Fold Cross Validation on initial training model
k_folds = KFold(n_splits = 10)

scores = cross_val_score(classifier, train_X, train_y, cv = k_folds)

print("Average CV Scores:", scores.mean())

#Grid Search with initial weights
param_grid = {'kernel': ['linear','poly','rbf', 'sigmoid'], 'C': [0.001, 0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(classifier, param_grid, cv=10)

grid_search.fit(test_X, test_y)

print("Best parameters:", grid_search.best_params_) 
print("Best cross-validation score:", grid_search.best_score_)

#Second Cross Validation with initial model and optimized hyperparameters
classifier2= svm.SVC(kernel='poly',C= 0.001, gamma = 10)
classifier2.fit(train_X, train_y)
k_folds = KFold(n_splits = 10)

scores2= cross_val_score(classifier2,train_X, train_y, cv = k_folds)

print("Average CV Scores:", scores2.mean())

#Evaluating Model on the Test Set
y_test_pred = classifier2.predict(test_X)
print("The accuracy on the test set is", metrics.accuracy_score(test_y, y_test_pred))

#%% Neural Network




#%% Evaluation






