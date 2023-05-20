# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 00:31:52 2022

@author: anand
"""

import os
import time
import random
import multiprocessing

import matplotlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.utils import shuffle
# from tensorboardX import SummaryWriter

# from keras.layers import Lambda, Input, Embedding, Dense, concatenate
# from keras.layers import Dropout, SpatialDropout1D, CuDNNLSTM, GaussianNoise
# from keras.models import Model
# from keras import initializers, regularizers, constraints, optimizers, layers
# from keras.optimizers import Adam
from tensorflow.keras import backend as K
# from keras.engine.topology import Layer, InputSpec
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
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from hyperopt import fmin, tpe, hp, anneal, Trials,SparkTrials
import mlflow
import pyspark

NEW_SEED = 2018
DEBUG_MODE = False

# Params

batch_size = 32 # how many samples to process at once

N_SPLITS = 5
N_EPOCHS = 5

# Load data


look_back=32
n_epochs=100
def create_shape(dataset,look_back):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)
  
alldata= pd.read_csv("TIRTL_hist_allclas_300s.csv")
tirtl=alldata[['TT','T_hist']]
tirtl.loc[:,'TT']=[ datetime.datetime.strptime(ss, '%Y/%m/%d %H:%M:%S') for ss in tirtl['TT']]

dyws=np.roll(tirtl['T_hist'].values,-42).reshape(int(len(tirtl)/int(1440/5)),int(1440/5)).astype('float64')
cs=dyws.mean(axis=1)>14
# dyws[~cs,:]=np.nan
dyws=dyws[cs,:]
datacln=dyws.reshape(int(1440/5)*dyws.shape[0],1)
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



def seed_keras(seed=NEW_SEED):
    """
    Function for setting a seed for reproduceability
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def local_cv(params):

    """
    This function gets a set of variable parameters in "param",
    sets param datatypes, loads data and runs the train_fn function.
    """

    param = {'L1': int(params['L1']),'L2': int(params['L2']),'look_back': int(params['look_back'])
      }

    # always call this before training for deterministic results
    seed_keras()
    

    val_score = train_fn(param, train,val)

    # IMPORTANT: Reset memory between testing models.
    K.clear_session()

   
    return val_score
	
	
def train_fn(param, train,val):
        """
        This is the main function that trains our model using K-fold,
        and ensembles the predictions of each fold to calcualte a
        final local F1 score.
        """
        # reshape into X=t and Y=t+1
        trainX, trainY = create_shape(train,param['look_back'])
        valX, valY = create_shape(val,param['look_back'])
        # testX, testY = create_shape(test)



        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
        # testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
        # Setting seed
        seed_keras()
    
        # model_params = {}
        # model_params['lstm_size'] = param['lstm_size'] # RNN layer hidden size
        # model_params['dense_dropout'] = param['dense_dropout'] # Dense layer dropout
    
        # Storing losses and rocaucs per epoch
    
        # a_val_losses = np.zeros((n_epochs))
    
        # Defining a tensorboardX writer
    
        # writer = SummaryWriter(get_run_name_from_params(model_params))
    
        # print("Model params: " + get_run_name_from_params(model_params))
    
        """
        Train meta will be our predictions on whole train set. This will,
        help in choosing the optimal threshold for the final predictions.
        """
    
        # train_meta = np.zeros(y_train.shape)
    
    
        model = new_model(
                param
                )
        start_time = time.time()
        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            min_delta=0.0001, 
            mode='min'
            )
# steps_per_epoch = len(trainX) // batch_size 
        # for i in range(rep):
        hist=model.fit(trainX,trainY, batch_size=32,epochs=n_epochs, callbacks=[custom_early_stopping],
          verbose=0, shuffle=True,validation_data=(valX,valY))

        """
        Here we store the validation losses per epoch so we can
        send them to the tensorboardX writer and observe the
        output.
        """
        avg_val_losses = []

        # for e in range(len(hist.history['val_loss'])):
        #     avg_val_losses.append(hist.history['val_loss'][e])

        end_time = time.time()
        elapsed_time = round((end_time-start_time), 1)


    
    
        # Writing to tensorboardX
    
        # for e in range(n_epochs):
        #     writer.add_scalar('avg_val_loss', avg_val_losses[e], e)
    
        # writer.add_scalar('final_rmse', final_thresh, 0)
    
        # print("final_f1: "+str(final_f1))
    
        # In this case we want to optimise F1.
        best_score=min(hist.history['val_loss'])
        print(" Val loss: {:.4f}".format(best_score) + " Time:{}s ".format(elapsed_time))   
        return best_score
    
def new_model(param):
    # Define a simple sequential model
      model = Sequential()
      model.add(LSTM(param['L1'], input_shape=(param['look_back'], 1), return_sequences=True))
      # ,stateful=True
      #model.add(LSTM(5, batch_input_shape=(batch_size, look_back, 1),  return_sequences=True))
      model.add(Dropout(0.2))
      model.add(LSTM(param['L2'], ))
      model.add(Dropout(0.2))
      model.add(Dense(1))
      model.compile(loss='mean_squared_error', optimizer='adam')
      return model