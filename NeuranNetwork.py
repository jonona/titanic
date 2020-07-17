#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:09:04 2020

@author: jonona
"""


from rf_lr import preprocess
import pandas as pd
import numpy as np

# Import necessary modules
from sklearn.model_selection import train_test_split

# Keras specific
#import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 




inputData = pd.read_csv("train.csv")
inputData = preprocess(inputData)

y = inputData.Survived.values
X = inputData.drop("Survived", axis=1).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

model = Sequential()
model.add(Dense(120, activation='relu', input_dim=37))
model.add(Dense(80, activation='relu'))
#model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20)

# Accuracy on training and validation sets

pred_train= model.predict(X_train)
scores = model.evaluate(X_train, y_train, verbose=0)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1]))   
 
pred_test= model.predict(X_val)
scores2 = model.evaluate(X_val, y_val, verbose=0)
print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))    

'''Test'''
inputData = pd.read_csv("test.csv")
id_col=pd.DataFrame(inputData["PassengerId"])
inputData = preprocess(inputData)

output = pd.DataFrame({'PassengerId': id_col["PassengerId"],'Survived': np.argmax(model.predict(inputData), axis=1)})
output.to_csv('submission3.csv', index=False)
