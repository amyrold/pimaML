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
from sklearn.model_selection import train_test_split


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





#%% Logistic Regresson




#%% SVM




#%% Neural Network




#%% Evaluation






