import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import pandas as pd
from pandas import read_csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Activation, Dropout
from numpy.testing import assert_allclose
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import joblib
import os
def create_shape(dataset,look_back):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)
  
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

alldata = pd.read_csv("TIRTL_hist_allclas_300s.csv")
tirtl = alldata[['TT', 'T_hist']]
tirtl.loc[np.logical_or(tirtl['T_hist']<5,tirtl['T_hist']>205),'T_hist']=-5
tirtl.loc[:, 'TT'] = [datetime.datetime.strptime(ss, '%Y/%m/%d %H:%M:%S') for ss in tirtl['TT']]
datacln=tirtl['T_hist'].values.reshape(-1,1)# fix random seed for reproducibility
#%%
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

def objective(trial):
            # Integer parameter (log)
        L1 = trial.suggest_int("L1", 128, 256, 4, False)
        L2 = trial.suggest_int("L2", 128, 256, 4, False)
        look_back = trial.suggest_int("look_back", 30, 120, 4, False)
        batch_size = trial.suggest_int("batch_size", 4, 128, 4, False)

        trainX, trainY = create_shape(train,look_back)
        valX, valY = create_shape(val,look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))
        model = Sequential()
        model.add(LSTM(L1, input_shape=(look_back, 1), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(L2, ))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            min_delta=0.0001, 
            mode='min'
            )
        hist=model.fit(trainX,trainY, batch_size=batch_size,epochs=80, callbacks=[custom_early_stopping],
          verbose=0, shuffle=True,validation_data=(valX,valY))
        best_score=min(hist.history['val_loss'])
        return best_score

def anal_objective(trial):
            # Integer parameter (log)
        L1 = trial.suggest_int("L1", 128, 256, 4, False)
        L2 = trial.suggest_int("L2", 128, 256, 4, False)
        look_back = trial.suggest_int("look_back", 30, 120, 4, False)
        batch_size = trial.suggest_int("batch_size", 4, 128, 4, False)


        trainX, trainY = create_shape(train,look_back)
        valX, valY = create_shape(val,look_back)
        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        valX = np.reshape(valX, (valX.shape[0], valX.shape[1], 1))

        model = Sequential()
        model.add(LSTM(L1, input_shape=(look_back, 1), return_sequences=True))
        # ,stateful=True
        #model.add(LSTM(5, batch_input_shape=(batch_size, look_back, 1),  return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(L2))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        custom_early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=20, 
            min_delta=0.0001, 
            mode='min',    restore_best_weights=False,
            )
        hist=model.fit(trainX,trainY, batch_size=batch_size,epochs=80, callbacks=[custom_early_stopping],
          verbose=2, shuffle=True,validation_data=(valX,valY))
        trainPredict = model.predict(trainX, batch_size=batch_size)
        # model.reset_states()
        valPredict = model.predict(valX, batch_size=batch_size)
        # model.reset_states()
        testPredict = model.predict(testX, batch_size=batch_size)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY1 = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        valPredict = scaler.inverse_transform(valPredict)
        valY1 = scaler.inverse_transform([valY])
        testY1 = scaler.inverse_transform([testY])
        trainScore = math.sqrt(mean_squared_error(trainY1[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        valScore = math.sqrt(mean_squared_error(valY1[0], valPredict[:,0]))
        testScore = math.sqrt(mean_squared_error(testY1[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        joblib.dump([trainScore, valScore, testScore, trainPredict, valPredict, testPredict],"allparam_studyanal.pkl")
