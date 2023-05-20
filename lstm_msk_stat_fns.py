import os
import time
import random
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.utils import shuffle
import pandas as pd
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Activation, Dropout,Masking
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import optuna
import joblib

def create_shape(dataset,look_back):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)
  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

alldata = pd.read_csv("TIRTL_hist_allclas_300s.csv")
tirtl = alldata[['TT', 'T_hist']]
tirtl.loc[np.logical_or(tirtl['T_hist']<5,tirtl['T_hist']>205),'T_hist']=-5
tirtl.loc[:, 'TT'] = [datetime.datetime.strptime(ss, '%Y/%m/%d %H:%M:%S') for ss in tirtl['TT']]
datacln=tirtl['T_hist'].values.reshape(-1,1)# fix random seed for reproducibility

np.random.seed(7)
# load the dataset
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(datacln)

# split into train and test sets
train_size = int(len(dataset) * (0.65))
val_size = int(len(dataset) * (0.15))
test_size = int(len(dataset) * (0.20))

train, val, test = dataset[0:train_size+1,:], dataset[train_size+1:train_size+val_size,:],dataset[-test_size:len(dataset),:]

[look_back,batch_size,L1]=[30,32,128]
# reshape into X=t and Y=t+1
trainX, trainY = create_shape(train,look_back)
valX, valY = create_shape(val,look_back)
testX, testY = create_shape(test,look_back)

lnT,lnV=len(trainX),len(valX)
trainX, trainY =   trainX[:-(lnT%batch_size),:], trainY[:-(lnT%batch_size)]
valX, valY =   valX[:-(lnV%batch_size),:], valY[:-(lnV%batch_size)]

#%%
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

        
def msk_stL():
        model = Sequential()
        model.add(Masking(mask_value=-5, input_shape=(look_back, 1)))
        model.add(LSTM(L1, input_shape=(look_back, 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(L1))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            min_delta=0.0001, 
            mode='min'
            )
        hist=model.fit(trainX,trainY, batch_size=batch_size,epochs=100, callbacks=[custom_early_stopping],
          verbose=0, shuffle=True,validation_data=(valX,valY))
        best_score=min(hist.history['val_loss'])
        return best_score


def msk_stF():
        model = Sequential()
        model.add(Masking(mask_value=-5, batch_input_shape=(batch_size,look_back, 1)))
        model.add(LSTM(L1, batch_input_shape=(batch_size,look_back, 1), return_sequences=True,stateful=True))
        model.add(Dropout(0.2))
        model.add(LSTM(L1,stateful=True))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            min_delta=0.0001, 
            mode='min'
            )
        hist=model.fit(trainX,trainY, batch_size=batch_size,epochs=100, callbacks=[custom_early_stopping],
          verbose=0, shuffle=True,validation_data=(valX,valY))
        best_score=min(hist.history['val_loss'])
        return best_score

def nomsk_stL():
        model = Sequential()
        # model.add(Masking(mask_value=-5, input_shape=(look_back, 1)))
        model.add(LSTM(L1, input_shape=(look_back, 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(L1))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            min_delta=0.0001, 
            mode='min'
            )
        hist=model.fit(trainX,trainY, batch_size=batch_size,epochs=100, callbacks=[custom_early_stopping],
          verbose=0, shuffle=True,validation_data=(valX,valY))
        best_score=min(hist.history['val_loss'])
        return best_score


def nomsk_stF():
        model = Sequential()
        # model.add(Masking(mask_value=-5, batch_input_shape=(batch_size,look_back, 1)))
        model.add(LSTM(L1, batch_input_shape=(batch_size,look_back, 1), return_sequences=True,stateful=True))
        model.add(Dropout(0.2))
        model.add(LSTM(L1,stateful=True))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            min_delta=0.0001, 
            mode='min'
            )
        hist=model.fit(trainX,trainY, batch_size=batch_size,epochs=100, callbacks=[custom_early_stopping],
          verbose=0, shuffle=True,validation_data=(valX,valY))
        best_score=min(hist.history['val_loss'])
        return best_score