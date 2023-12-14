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
# * Dummy Classifier - Aaron
# Evaluation - Aaron

#%%
import os
import pandas as pd
from IPython.display import display



from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm #importing svm model
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier


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
train_X, test_X, train_y, test_y = train_test_split(X_norm, Y, test_size = 0.20, shuffle = True)


#%% Model building
#%% KNN

knn = KNeighborsClassifier()

#fit knn on training set
knn_og = knn.fit(train_X, train_y)


#N-fold cross validation to evaluate weights
kf = KFold(n_splits = 10)
scores = cross_val_score(knn_og, train_X, train_y, cv = kf)
print(f"Accuracy Scores Default Hyperparams: {scores}")
knn_baseline = scores.mean()
print(f"Average Accuracy Scores: {knn_baseline}")


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
knn_hp = knn_gs_cv.best_params_
print(f"best_params: {knn_hp}")


#doing k cross validation with new tuned model on the training set to see how the accuracies change
scores_new = cross_val_score(knn_gs_cv, train_X, train_y, cv = kf)
knn_tuned = scores_new.mean()

#print accuracy scores
print(f"Accuracy Scores after Tuning: {scores_new}")
print(f"Average Accuracy Scores after Tuning: {knn_tuned}")

knn_test = knn_gs_cv.score(test_X,test_y)
print(f"Accuracy after scoring tuned model on test: {knn_test}")

#%% Logistic Regresson


model= LogisticRegression()
model_og = model.fit(train_X, train_y)


#evaluates the model with default parameters using 10-fold cross validation
kfold = KFold(n_splits=10)
kfold.get_n_splits(train_X)
results = cross_val_score(model, train_X, train_y, cv=kfold)

# Output the accuracy. Calculate the mean and std across all folds.
print(f"LR_baseline: {results.mean()*100.0}%")
LR_baseline = results.mean()

 #grid search
hp = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'solver': ['liblinear','lbfgs', 'newton-cg','sag','saga', 'newton-cholesky']}
gs = GridSearchCV(estimator = model, param_grid = hp, cv = kfold)
gs.fit(train_X, train_y)
print(f"LR_hp: {gs.best_params_}")
LR_hp = gs.best_params_

#trains model using new hyperparameters
scores_new = cross_val_score(gs, train_X, train_y, cv = kfold)
print(f'LR_tuned: {scores_new.mean()*100}%')
LR_tuned = scores_new.mean()

#evaluates on test dataset
accuracy_score_test = gs.score(test_X, test_y)

print(f'Logistic regression Accuracy score on the test set post grid search: {accuracy_score_test*100}%')

print(f'LR_test: {accuracy_score_test*100}%')
LR_test = accuracy_score_test

 #%% SVM
#fit model for classification
#Creating SVM classifier with linear kernel
classifier = svm.SVC(kernel='linear') 

#Fitting the model with the training sets
classifier.fit(train_X,train_y)

train_pred = classifier.predict(train_X) 

#K-Fold Cross Validation
k_folds = KFold(n_splits = 10)

scores = cross_val_score(classifier, train_X, train_y, cv = k_folds)

SVM_baseline = scores.mean()
print("Average SVM CV Scores with Default Hyperparameters:", SVM_baseline)

#Grid Search
param_grid = {'kernel': ['linear','poly'],'C':[.001, 0.1, 1, 10], 'gamma': [0.001, 0.1, 1, 10]}

grid_search = GridSearchCV(classifier, param_grid, cv=10)

grid_search.fit(train_X, train_y)

SVM_hp = grid_search.best_params_
print("SVM Optimal Hyperparameters:", SVM_hp)

#Second Cross Validation with initial model and optimized hyperparameters
k_folds = KFold(n_splits = 10)
scores2= cross_val_score(grid_search,train_X, train_y, cv = k_folds)
SVM_tuned = scores2.mean()
print("SVM Average CV Scores with Optimal Hyperparameters:", SVM_tuned)

#Evaluating Model on the Test Set
svm_pred = grid_search.predict(test_X)
SVM_test = metrics.accuracy_score(test_y, svm_pred)
print("Final SVM Accuracy on the Test Set:", SVM_test)

#%% Neural Network

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

#fit mlp on training set
mlp_og = mlp.fit(train_X, train_y)

#N-fold cross validation to evaluate weights
kf = KFold(n_splits = 10)
kf.get_n_splits(train_X)
mlp_baseline = cross_val_score(mlp_og, train_X, train_y, cv = kf)
mlp_baseline = mlp_baseline.mean()
print(f"Accuracy Scores Default Hyperparams: {mlp_baseline}")
print(f"Average CV Accuracy Scores on OG model: {mlp_baseline.mean()}")


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

#fitting the model with best hyper parameters on the training set
mlp_gs_cv.fit(train_X, train_y)

#Find best parameters from the Grid search CV
print(f"best_params: {mlp_gs_cv.best_params_}")
mlp_hp = mlp_gs_cv.best_params_

#doing k cross validation with new tuned model on the training set to see how the accuracies change
mlp_tuned = cross_val_score(mlp_gs_cv, train_X, train_y, cv = kf)
mlp_tuned = mlp_tuned.mean()

#print accuracy scores after hyper parameter tuning
print(f"Accuracy Scores after Tuning: {mlp_tuned}")
print(f"Average Accuracy Scores after Tuning: {mlp_tuned.mean()}")

#print accuracy on test set
mlp_test = mlp_gs_cv.score(test_X, test_y)
print(f"The accuracy of the MLP on the test score is: {mlp_test}")

#%% Dummy Classifier
# Create dummy classifier with most frequest
dumb_MF = DummyClassifier(strategy= 'most_frequent')
dumb_MF.fit(train_X,train_y)

#%% Evaluation

# Classification Reports
knn_pred = knn_gs_cv.predict(test_X)
knn_cr = classification_report(test_y, knn_pred)
print('KNN Classification Report:\n', knn_cr)

LR_pred = gs.predict(test_X)
LR_cr = classification_report(test_y, LR_pred)
print('LR Classification Report:\n', LR_cr)

svm_pred = grid_search.predict(test_X)
svm_cr = classification_report(test_y, svm_pred)
print('SVM Classification Report:\n', svm_cr)

mlp_pred = mlp_gs_cv.predict(test_X)
mlp_cr = classification_report(test_y, mlp_pred)
print('mlp Classification Report:\n', mlp_cr)

# Create classification report for dummy MF
DM_pred = dumb_MF.predict(test_X)
DM_cr = classification_report(test_y, DM_pred)
print('DM Classification Report:\n:', DM_cr)

# Accuracy Score Table
data = [[knn_baseline, knn_tuned,knn_test],
        [LR_baseline, LR_tuned,LR_test],
        [SVM_baseline, SVM_tuned,SVM_test],
        [mlp_baseline, mlp_tuned,mlp_test]]

columns = ['baseline','tuning','test']
index = ['KNN','LR','SVM','MLP']

score_grid = pd.DataFrame(data, columns=columns, index=index)
display(score_grid)

