# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:49:32 2018

@author: 009322
"""

import numpy as np
import pandas as pd
import matplotlib as plt
import datetime
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
 
Total_Sample = pd.read_csv('Total_Sample_r.csv')
Total_Sample['DATE'] = pd.to_datetime(Total_Sample['DATE'])
DATES = pd.DataFrame(Total_Sample['DATE'].drop_duplicates())
DATES = DATES.reset_index(drop = True)
Feature = Total_Sample.columns[9:22]
Feature =Feature.append(Total_Sample.columns[24:55])

for month in range(0,21):
    start = DATES.iloc[month,0]
    re_start = DATES.iloc[month+28,0]
    finish = DATES.iloc[month+33,0]
    test = DATES.iloc[month+36,0]
    
    Recent_Sample  = Total_Sample[(Total_Sample['DATE'] >= re_start) 
            & (Total_Sample['DATE'] <= finish)] 
    Recent_Sample = Recent_Sample.drop(['Unnamed: 0'], axis = 1)
    Train_Sample = Total_Sample[(Total_Sample['DATE'] >= start) 
            & (Total_Sample['DATE'] <= finish)]
    Train_Sample = Train_Sample.drop(['Unnamed: 0'], axis = 1)
    Train_Sample = Train_Sample.append(Recent_Sample)
    Test_Sample = Total_Sample[Total_Sample['DATE'] == test] 
    Test_Sample = Test_Sample.drop(['Unnamed: 0'], axis = 1)
    
    Train_Feature = Train_Sample[Feature]
    Train_Target = np.ravel(Train_Sample[['PERFORMANCE']]) 
    Test_Feature = Test_Sample[Feature]
    Test_Target = np.ravel(Test_Sample[['PERFORMANCE']])
    
    criteria = 0
    weight = 0.5
    count = 1
    while criteria < 30 or criteria > 50:
        forest = RandomForestClassifier(n_estimators = 200,
                                        max_depth = 15,
                                        min_samples_split = 50,
                                        min_samples_leaf = 20,
                                        class_weight = {True:weight, False:1-weight},
                                        random_state = 1,oob_score = True)
    
        Train_Forest = forest.fit(Train_Feature, Train_Target)
        Predictor_Forest = forest.predict(Test_Feature)
        
        criteria =sum(Predictor_Forest)
        if criteria > 50:
            weight -= 0.01
        elif criteria < 30:
            weight += 0.01
        count += 1
        if count >= 40:
            break
        
        importance = Train_Forest.feature_importances_
        Score = pd.DataFrame(index = Feature)
        Score['IMPORTANCE'] = importance
        key = Score['IMPORTANCE'].idxmax()
        
    Predict = pd.DataFrame({'Predict':Predictor_Forest}, 
                           index = Test_Sample.index)
    Result = pd.concat([Test_Sample, Predict], axis=1, 
                       join_axes=[Test_Sample.index])
    for col in Total_Sample.columns[24:55]:
        Result = Result.drop([col], axis = 1)
    Result.to_csv('Result_rf_'+ str(month)+ str(key) +'.csv')

'''

forest = RandomForestClassifier(n_estimators = 200,
                            max_depth = 15,
                            min_samples_split = 50,
                            min_samples_leaf = 20,
                            random_state = 1,oob_score = True)

Train_Forest = forest.fit(Train_Feature, Train_Target)
Predictor_Forest = forest.predict(Test_Feature)
Predict = pd.DataFrame({'Predict':Predictor_Forest}, index = Test_Sample.index)
Result = pd.concat([Test_Sample, Predict], axis=1, join_axes=[Test_Sample.index])

for col in Total_Sample.columns[24:55]:
    Result = Result.drop([col], axis = 1)
Result.to_csv('Result_r_201709.csv')

print(Train_Forest.score(Train_Feature, Train_Target))
print(forest.oob_score_)
print(Train_Forest.score(Test_Feature, Test_Target))
print('  ')

importance = Train_Forest.feature_importances_
Score = pd.DataFrame(index = Feature)
Score['IMPORTANCE'] = importance
print(Score)
'''