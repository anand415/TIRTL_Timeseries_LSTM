
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
from  lstm_stgs_drp_fns import *
import pickle
import joblib
bestscore_stg4_nodrp=[]
bestscore_stg3_nodrp=[]
bestscore_stg2_nodrp=[]
bestscore_stg1_nodrp=[]
bestscore_stg4_drp2=[]
bestscore_stg3_drp2=[]
bestscore_stg2_drp2=[]
bestscore_stg1_drp2=[]
bestscore_stg4_drp6=[]
bestscore_stg3_drp6=[]
bestscore_stg2_drp6=[]
bestscore_stg1_drp6=[]
for ii in range(1,40):
   print(ii,'nodrp') 
   # bestscore_stg1_nodrp.append(stg1_nodrp())
   # bestscore_stg2_nodrp.append(stg2_nodrp())
   # bestscore_stg3_nodrp.append(stg3_nodrp())
   # bestscore_stg4_nodrp.append(stg4_nodrp())

   print(ii,'drp2') 
   bestscore_stg4_drp2.append(stg4_drp2())
   bestscore_stg3_drp2.append(stg3_drp2())
   bestscore_stg2_drp2.append(stg2_drp2())
   bestscore_stg1_drp2.append(stg1_drp2())

   print(ii,'drp6') 
   bestscore_stg4_drp6.append(stg4_drp6())
   bestscore_stg3_drp6.append(stg3_drp6())
   bestscore_stg2_drp6.append(stg2_drp6())
   bestscore_stg1_drp6.append(stg1_drp6())

   allV=[bestscore_stg4_nodrp,bestscore_stg3_nodrp,bestscore_stg2_nodrp,
   bestscore_stg1_nodrp,bestscore_stg4_drp2,bestscore_stg3_drp2,
   bestscore_stg2_drp2,bestscore_stg1_drp2,bestscore_stg4_drp6,bestscore_stg3_drp6,bestscore_stg2_drp6,bestscore_stg1_drp6]
   joblib.dump(allV, "stg_drp.pkl")
