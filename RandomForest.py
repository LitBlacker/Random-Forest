# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:42:57 2017

@author: 009322
"""

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

Total_Sample = pd.read_csv('Total_Sample_n_q.csv')
Total_Sample['DATE'] = pd.to_datetime(Total_Sample['DATE'])
Train_Sample = Total_Sample[(Total_Sample['DATE'] > '2014/09/01') 
    & (Total_Sample['DATE'] < '2017/06/01')]

Train_Sample = Train_Sample.drop(['Unnamed: 0'], axis = 1)
Test_Sample = Total_Sample[(Total_Sample['DATE'] > '2017/09/01') 
    & (Total_Sample['DATE'] < '2017/10/01')]

Test_Sample = Test_Sample.drop(['Unnamed: 0'], axis = 1)
#print(Test_Sample.head())

Feature = ['ROA', 'EP', 'SP', 'BP', 'VOLATILITY', 'DRAWDOWN', 'REV', 'MV', 
           'EV', 'VOLUME', 'EVMV']

Train_Feature = Train_Sample[Feature]
Train_Target = np.ravel(Train_Sample[['PERFORMANCE']])

Test_Feature = Test_Sample[Feature]
Test_Target = np.ravel(Test_Sample[['PERFORMANCE']])

forest = RandomForestClassifier(n_estimators = 100,
                                max_depth = 25,
                                min_samples_split = 50,
                                random_state = 1,oob_score = True)

Train_Forest = forest.fit(Train_Feature, Train_Target)
Predictor_Forest = forest.predict(Test_Feature)
Predict = pd.DataFrame({'Predict':Predictor_Forest}, index = Test_Sample.index)
Result = pd.concat([Test_Sample, Predict], axis=1, join_axes=[Test_Sample.index])
Result.to_csv('Result_n_q_201709.csv')


print(Train_Forest.score(Train_Feature, Train_Target))
print(forest.oob_score_)
print(Train_Forest.score(Test_Feature, Test_Target))
print('  ')

importance = Train_Forest.feature_importances_
for i in range(0,11):
    print(importance[i])