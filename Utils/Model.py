
import pandas as pd
import numpy as np

import pickle
import sys
import os

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def model_building(X, y, crossfolds, model):
    scores = []
    names = []
    for i, j in model:
        cv_scores = cross_val_score(j, X, y, cv=crossfolds)
        scores.append(cv_scores)
        names.append(i)
    
    for k in range(len(scores)):
        print(names[k], scores[k].mean())
    
    return

# Blending

def models_blending_list():
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('GBR', GradientBoostingRegressor()))
    return models

def model_blending(models, X_train, X_val, y_train, y_val):
    level1_predictions = []
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred = y_pred.reshape(len(y_pred), 1)
        level1_predictions.append(y_pred)
        # print(y_pred)
        # print(level1_predictions)

    level1_predictions = np.hstack(level1_predictions)
    # print(level1_predictions)
    level2_predictions = MLPRegressor()
    level2_predictions.fit(level1_predictions, y_val)
    return level2_predictions

def model_blending_predictions(models, blending, X_test):
    models_X = []
    for name, model in models:
        y_predictions = model.predict(X_test)
        y_predictions = y_predictions.reshape(len(y_predictions), 1)
        models_X.append(y_predictions)
    
    models_X = np.hstack(models_X)

    return blending.predict(models_X)