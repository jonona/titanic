#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:36:26 2020

@author: jonona
"""


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import MinMaxScaler

def as_labels(data):
     for col in data.columns:
         if pd.api.types.is_string_dtype(data[col]):
            lbl = LabelEncoder()
            lbl.fit(list(data[col].values))
            data[col] = lbl.transform(list(data[col].values))
            data[col] = data[col].astype(object)
     return data

def preprocess(data):
    #scaler = MinMaxScaler() 
    
    data["Cabin"] = data["Cabin"].fillna("U0")
    data["Age"] = data.groupby("Pclass")["Age"].transform(lambda x: x.fillna(x.median()))
    data["Fare"] = data.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["SibSp"] = data["SibSp"].fillna(0)
    data["Parch"] = data["Parch"].fillna(0)
    
    #data["FamilySize"] = data["SibSp"]+data["Parch"]+1
    #data["IsAlone"]=np.where(data["FamilySize"]==1,1,0)
    data['Cabin'] = data['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    #data["Fare"] = np.log1p(data["Fare"])
    
    
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr','Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    data["Title"] = data["Title"].fillna("None")
    
    data["Pclass"]=data["Pclass"].astype(object)
    data['Fare'] = pd.qcut(data['Fare'], 13, duplicates='drop')
    data['Age'] = pd.qcut(data['Age'], 10, duplicates='drop')
    data['Ticket_Frequency'] = data.groupby('Ticket')['Ticket'].transform('count')
    
    # data.loc[:,"Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1)) 
    # data.loc[:,"Fare"] = scaler.fit_transform(data["Fare"].values.reshape(-1,1)) 
    
    data = data.drop(['Name','Ticket','PassengerId','Cabin'], axis = 1)
    data = as_labels(data)
    data=pd.get_dummies(data)
    data = data.drop(['Sex_0'], axis = 1)
    
    return data


# inputData = pd.read_csv("train.csv")
# inputData = preprocess(inputData)

# y = inputData.Survived
# X = inputData.drop("Survived", axis=1)

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)

# kf = KFold(n_splits=5, random_state=42, shuffle=True)

# print("Linear Regression")
# lr = LogisticRegression(class_weight = 'balanced', solver = 'liblinear',penalty="l2")
# lr.fit(X_train,y_train)
# print(cross_val_score(lr, X_train,y_train, cv=kf).mean())
# print("Score on train data: {:.05f} \nScore on test data: {:.05f}".format(lr.score(X_train,y_train), lr.score(X_val, y_val)))

# print("Random Forests")
# rf = RandomForestClassifier(criterion='gini',
#                             n_estimators=1750,
#                             max_depth=7,
#                             min_samples_split=6,
#                             min_samples_leaf=6,
#                             max_features='auto',
#                             oob_score=True,
#                             random_state=42,
#                             n_jobs=-1,
#                             verbose=1)
# rf.fit(X_train,y_train)
# print(cross_val_score(rf, X_train,y_train, cv=kf).mean())
# print("Score on train data: {:.05f} \nScore on test data: {:.05f}".format(rf.score(X_train,y_train), rf.score(X_val, y_val)))

# feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
# feat_importances.nlargest(30).plot(kind='barh',figsize=(30,10))


# '''Test'''
# inputData = pd.read_csv("test.csv")
# id_col=pd.DataFrame(inputData["PassengerId"])
# inputData = preprocess(inputData)

# output = pd.DataFrame({'PassengerId': id_col["PassengerId"],'Survived': lr.predict(inputData)})
# output.to_csv('submission1.csv', index=False)

# output = pd.DataFrame({'PassengerId': id_col["PassengerId"],'Survived': rf.predict(inputData)})
# output.to_csv('submission2.csv', index=False)


