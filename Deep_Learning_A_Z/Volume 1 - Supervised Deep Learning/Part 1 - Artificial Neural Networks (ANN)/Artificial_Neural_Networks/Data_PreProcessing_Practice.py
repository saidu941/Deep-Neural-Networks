#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:24:02 2017

@author: saidu941
"""

#importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Importing the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:-1].values
Y=dataset.iloc[:,13].values



#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]



#Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split;
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)




#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)







#MAKING the ANN Model
import keras 
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the first input layer and the hidden layer
classifier.add(Dense(units=6, activation= 'relu', kernel_initializer='uniform', input_dim = 11 ))

#Adding the second hidden layer
classifier.add(Dense(units=6, activation= 'relu', kernel_initializer='uniform'))

#Adding the output layer
classifier.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])




#FITTING THE ANN TO THE TRAINING SET
classifier.fit(X_train, Y_train, batch_size=10, epochs=10)










#MAKING THE PREDICTIONS AND EVALUATING THE MODEL
#predicting the test set results
Y_pred= classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)












