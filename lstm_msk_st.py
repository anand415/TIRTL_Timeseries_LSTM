
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
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from  lstm_msk_stat_fns import *
import pickle
import joblib
bestscore_msk_stL=[]
bestscore_msk_stF=[]
bestscore_nomsk_stL=[]
bestscore_nomsk_stF=[]
for ii in range(1,2):
   bestscore_msk_stL.append(msk_stL())
   bestscore_msk_stF.append(msk_stF())
   bestscore_nomsk_stL.append(nomsk_stL())
   bestscore_nomsk_stF.append(nomsk_stF())
   joblib.dump([bestscore_msk_stL,bestscore_msk_stF,bestscore_nomsk_stL,bestscore_nomsk_stF], "msk_stlf.pkl")
