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
pima = pd.read_csv('/Users/nirushanbhag/Downloads/ML_project/pimaML/diabetes.csv')

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


#%% Model building




#%% KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

knn = KNeighborsClassifier()

#fit knn on training set
knn_og = knn.fit(train_X, train_y)


#N-fold cross validation to evaluate weights
kf = KFold(n_splits = 10)
scores = cross_val_score(knn_og, train_X, train_y, cv = kf)
print(f"Accuracy Scores Default Hyperparams: {scores}")
print(f"Average Accuracy Scores: {scores.mean()}")


#setting a grid
grid = {
'n_neighbors': [1,2,3,4,5,6,7,8,9,10],
'weights': ['uniform', 'distance'],
'p': [1,2]
}


#performing grid search cv, use mlp as the base model
knn_gs_cv = GridSearchCV(estimator=knn, param_grid=grid, cv= kf) #returns a model with the best hyperparameters


#fitting the model with best hyper parameters on the training set
knn_gs_cv.fit(train_X, train_y)
#printing the model’s best parameters
print(f"best_params: {knn_gs_cv.best_params_}")


#doing k cross validation with new tuned model on the training set to see how the accuracies change
scores_new = cross_val_score(knn_gs_cv, train_X, train_y, cv = kf)

#print accuracy scores
print(f"Accuracy Scores after Tuning: {scores_new}")
print(f"Average Accuracy Scores after Tuning: {scores_new.mean()}")


#knn_test = KNeighborsClassifier(n_neighbors=9, p=1, weights='distance')
print("Accuracy after scoring tuned model on test", knn_gs_cv.score(test_X,test_y))

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
from sklearn.model_selection import KFold


kf = KFold(n_splits=10)
kf.get_n_splits(train_X)


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

#fit mlp on training set
mlp_og = mlp.fit(train_X, train_y)

#N-fold cross validation to evaluate weights
kf = KFold(n_splits = 10)
scores = cross_val_score(mlp_og, train_X, train_y, cv = kf)
print(f"Accuracy Scores Default Hyperparams: {scores}")
print(f"Average CV Accuracy Scores on OG model: {scores.mean()}")

#setting a grid
grid = {
    'hidden_layer_sizes': [(100,100), (100,100,50)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'lbfgs'],
    # 'alpha': [0.0001, 0.05],
    # 'learning_rate': ['constant','adaptive'],
    'max_iter' : [10000]
}

#performing grid search cv
mlp_gs_cv = GridSearchCV(estimator=mlp, param_grid=grid, cv= kf) #returns a model with the best hyperparameters

#Find best parameters from the Grid search CV
print(f"best_params: {mlp_gs_cv.best_params_}")

#fitting the model with best hyper parameters on the training set
mlp_gs_cv.fit(train_X, train_y)

#doing k cross validation with new tuned model on the training set to see how the accuracies change
scores_new = cross_val_score(mlp_gs_cv, train_X, train_y, cv = kf)

#print accuracy scores
print(f"Accuracy Scores after Tuning: {scores_new}")
print(f"Average Accuracy Scores after Tuning: {scores_new.mean()}")

#print accuracy on test set
accuracy_score_test = mlp_gs_cv.score(test_X, test_y)
print(f"The accuracy of the MLP on the test score is: {accuracy_score_test}")

#%% Evaluation






