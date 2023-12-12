#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:38:00 2023

@author: blue
"""

# GridSearch Tutorial: https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

hp = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'solver': ['lbfgs', 'newton-cg','sag','saga', 'newton-cholesky']}

grid = GridSearchCV(LogisticRegression(), hp, refit = True, verbose = 3)

grid.fit(train_X, train_Y)
grid.best_params_

pred_grid = grid.predict(dev_X)
c_GRID = classification_report(dev_Y, pred_grid)
print(c_GRID)

# Tweaking the hyper parameters lead to a slight decrease in accuracy and an increase in recall