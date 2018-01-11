# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:42:57 2017

@author: 009322
"""

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

Total_Sample = pd.read_csv('Total_Sample_rp_q.csv')
Total_Sample['DATE'] = pd.to_datetime(Total_Sample['DATE'])
Train1 = Total_Sample[(Total_Sample['DATE'] > '2013/04/01') 
    & (Total_Sample['DATE'] < '2016/01/01')]
Train1 = Train1.drop(['Unnamed: 0'], axis = 1)

Test1 = Total_Sample[(Total_Sample['DATE'] > '2016/04/01') 
    & (Total_Sample['DATE'] < '2016/05/01')]
Test1 = Test1.drop(['Unnamed: 0'], axis = 1)

Feature = ['ROA', 'EP', 'SP', 'BP', 'VOLATILITY', 'DRAWDOWN', 'REV', 'MV', 
           'EV', 'VOLUME', 'EVMV']

Train1_Feature = Train1[Feature]
Train1_Target = np.ravel(Train1[['MRP']])

Test1_Feature = Test1[Feature]
Test1_Target = np.ravel(Test1[['MRP']])

forest1 = RandomForestClassifier(n_estimators = 100,
                                 min_samples_split = 100,
                                 min_samples_leaf = 10,
                                 random_state = 1,oob_score = True)

Classifier1 = forest1.fit(Train1_Feature, Train1_Target)
Test1_Predict = Classifier1.predict(Test1_Feature)

Test_Class1 = pd.DataFrame({'CLASS1':Test1_Predict}, index = Test1.index)
Test2 = pd.concat([Test1, Test_Class1], axis=1, join_axes=[Test1.index])
Test2 = Test2.where(Test2['CLASS1'] == True)
Test2 = Test2.dropna(axis=0, how='any')
Test2['TRP'] = Test2['TRP'] == 1


Train2_Feature = Train1[Feature]
Train2_Target = np.ravel(Train1[['TRP']])

Test2_Feature = Test2[Feature]
Test2_Target = np.ravel(Test2[['TRP']])

forest2 = RandomForestClassifier(n_estimators = 100,
                                 min_samples_split = 10,
                                random_state = 1,oob_score = True)

Classifier2 = forest2.fit(Train2_Feature, Train2_Target)
Test2_Predict = Classifier2.predict(Test2_Feature)
Test_Class2 = pd.DataFrame({'CLASS2':Test2_Predict}, index = Test2.index)
Result =  pd.concat([Test2, Test_Class2], axis=1, join_axes=[Test2.index])
Result.to_csv('Result_n_rp_201604.csv')


print(Classifier1.score(Train1_Feature, Train1_Target))
print(forest1.oob_score_)
print(Classifier1.score(Test1_Feature, Test1_Target))
print(' ')
print(Classifier2.score(Train2_Feature, Train2_Target))
print(forest2.oob_score_)
print(Classifier2.score(Test2_Feature, Test2_Target))
print(' ')
importance = Classifier2.feature_importances_
for i in range(0,11):
    print(importance[i])
