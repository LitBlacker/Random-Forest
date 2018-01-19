# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:49:32 2018

@author: 009322
"""

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
 
Total_Sample = pd.read_csv('Total_Sample_r.csv')
Total_Sample['DATE'] = pd.to_datetime(Total_Sample['DATE'])
Train_Sample = Total_Sample[(Total_Sample['DATE'] > '2013/01/01') 
            & (Total_Sample['DATE'] < '2015/11/01')]
 
Train_Sample = Train_Sample.drop(['Unnamed: 0'], axis = 1)
Test_Sample = Total_Sample[(Total_Sample['DATE'] > '2016/01/01') 
            & (Total_Sample['DATE'] < '2016/02/01')]
 
Test_Sample = Test_Sample.drop(['Unnamed: 0'], axis = 1)


Feature = Total_Sample.columns[9:22]
Feature =Feature.append(Total_Sample.columns[24:55])

Train_Feature = Train_Sample[Feature]
Train_Target = np.ravel(Train_Sample[['PERFORMANCE']])
 
Test_Feature = Test_Sample[Feature]
Test_Target = np.ravel(Test_Sample[['PERFORMANCE']])

forest = RandomForestClassifier(n_estimators = 200,
 #                               max_depth = 15,
 #                               min_samples_split = 50,
                                min_samples_leaf = 20,
                                class_weight = {True:0.48, False:0.52},
                                random_state = 1,oob_score = True)

Train_Forest = forest.fit(Train_Feature, Train_Target)
Predictor_Forest = forest.predict(Test_Feature)
Predict = pd.DataFrame({'Predict':Predictor_Forest}, index = Test_Sample.index)
Result = pd.concat([Test_Sample, Predict], axis=1, join_axes=[Test_Sample.index])

for col in Total_Sample.columns[24:55]:
    Result = Result.drop([col], axis = 1)
print(Result.info())

Result.to_csv('Result_r_201601.csv')

print(Train_Forest.score(Train_Feature, Train_Target))
print(forest.oob_score_)
print(Train_Forest.score(Test_Feature, Test_Target))
print('  ')

importance = Train_Forest.feature_importances_
Score = pd.DataFrame(index = Feature)
Score['IMPORTANCE'] = importance
print(Score)
