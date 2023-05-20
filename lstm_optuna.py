import os
import time
import random
import multiprocessing
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle
import pandas as pd
from pandas import read_csv
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, Activation, Dropout
from numpy.testing import assert_allclose
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from  opurtuna_lstm_fns import *
import joblib
import optuna

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


study = optuna.create_study(study_name='allparam',directions=["minimize"])
study.optimize(objective, n_trials=5)
joblib.dump(study, "allparam.pkl")

for ii in range(0,10):
  study = joblib.load("allparam.pkl")
  # study = optuna.load_study(study_name="LSTMS")
  # study = optuna.create_study(study_name='LSTMS', load_if_exists=True,directions=["minimize"])
  study.optimize(objective, n_trials=10)
  joblib.dump(study, "allparam.pkl")


study = joblib.load("allparam.pkl")
anal_objective(study.best_trial)  # calculate acc, f1, recall, and precision
# new_model = tf.keras.models.load_model('saved_model/my_model')
[trainScore, valScore, testScore, trainPredict, valPredict, testPredict] = joblib.load("allparamanal.pkl")