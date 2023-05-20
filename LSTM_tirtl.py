# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:05:14 2022

@author: anand
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import math
import tensorflow as tf
# from tensorflow.compat.v1.keras.layers import (
#     CuDNNLSTM as LSTM,
# )
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Activation, Dropout
from numpy.testing import assert_allclose
import datetime

# Load the TensorBoard notebook extension
# load_ext tensorboard


alldata= pd.read_csv("TIRTL_hist_allclas_300s.csv")
tirtl=alldata[['TT','T_hist']]
tirtl.loc[:,'TT']=[ datetime.datetime.strptime(ss, '%Y/%m/%d %H:%M:%S') for ss in tirtl['TT']]
#%%
# dy=tirtl.shift['datetime']
dyws=np.roll(tirtl['T_hist'].values,-42).reshape(int(len(tirtl)/int(1440/5)),int(1440/5)).astype('float64')
cs=dyws.mean(axis=1)>14
# dyws[~cs,:]=np.nan
dyws=dyws[cs,:]
datacln=dyws.reshape(int(1440/5)*dyws.shape[0],1)

#%%
# convert an array of values into a dataset matrix
def create_shape(dataset):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)
#%%
[look_back,batch_size,L1,L2,epchs,rep]=[30,32,128,16,100,1]

# fix random seed for reproducibility
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

# reshape into X=t and Y=t+1
trainX, trainY = create_shape(train)
# trainX, trainY =   trainX[:-(len(trainX)%batch_size),:], trainY[:-(len(trainX)%batch_size)]
# print(trainX.shape)

valX, valY = create_shape(val)
testX, testY = create_shape(test)

  #%%
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# train_data = tf.data.Dataset.from_tensor_slices((trainX, trainY))
# train_data = train_data.repeat().batch(batch_size, drop_remainder=True)

  
  # Define a simple sequential model
model = Sequential()
model.add(LSTM(L1, input_shape=(look_back, 1),  return_sequences=True))
# ,stateful=True
#model.add(LSTM(5, batch_input_shape=(batch_size, look_back, 1),  return_sequences=True))
# model.add(LSTM(L2, ))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


steps_per_epoch = len(trainX) // batch_size 
# print(steps_per_epoch,trainX.shape)
for i in range(rep):
  model.fit(trainX,trainY, epochs=30,
  verbose=2, shuffle=False,validation_data=(valX,valY))
  model.reset_states()
# shutil.rmtree("training_2}", ignore_errors=True)
# shutil.rmtree('logs', ignore_errors=True)
# Include the epoch in the file name (uses `str.format`)
# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
