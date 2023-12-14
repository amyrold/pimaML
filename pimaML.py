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
classifier = svm.SVC(kernel='linear') 

#Fitting the model with the training sets
classifier.fit(train_X, train_y)

#Predict the response for training dataset
y_train_pred = classifier.predict(train_X)

#Evaluating Accuracy of training model on training model
print("### Evaluating Accuracy of SVM classifier on training set ###")
print("Accuracy of SVM:",metrics.accuracy_score(train_y, y_train_pred))

#Evaluating Accuracy of training model on development set

#Predict the response for test dataset
y_dev_pred = classifier.predict(test_X)
print("\n### Evaluating Accuracy of SVM classifier on development set ###")
print("Accuracy of SVM:",metrics.accuracy_score(test_y, y_dev_pred))




#%% Neural Network

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

#fit mlp on training set
mlp_og = mlp.fit(train_X, train_y)

#N-fold cross validation to evaluate weights
kf = KFold(n_splits = 10)
scores = cross_val_score(mlp_og, train_X, train_y, cv = kf)
print(f"Accuracy Scores Default Hyperparams: {scores}")
print(f"Average Accuracy Scores: {scores.mean()}")

#setting a grid
grid = {
    'hidden_layer_sizes': [(100,100), (100,100,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'lbfgs', 'sgd'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    'max_iter' : [10000]
}

#performing grid search cv
mlp_gs_cv = GridSearchCV(estimator=mlp, param_grid=grid, cv= kf) #returns a model with the best hyperparameters

#fitting the model with best hyper parameters on the training set
mlp_gs_cv.fit(train_X, train_y)

#doing k cross validation with new tuned model on the training set to see how the accuracies change
scores_new = cross_val_score(mlp_gs_cv, train_X, train_y, cv = kf)

#print accuracy scores
print(f"Accuracy Scores after Tuning: {scores_new}")
print(f"Average Accuracy Scores after Tuning: {scores_new.mean()}")

#%% Evaluation






