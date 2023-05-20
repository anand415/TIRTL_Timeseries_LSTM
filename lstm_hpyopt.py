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
from  hyperopt_lstm_fns import *
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

matplotlib.use('agg')
plt.close('all')

# Seed and debug mode option


# convert an array of values into a dataset matrix
#%%
# dy=tirtl.shift['datetime']

if __name__ == "__main__":
       
    # Parameter space for hyperopt to explore
        SPACE = {'L1': hp.quniform('L1', 16, 128, 16),
                 'L2': hp.quniform('L2', 8, 128, 8),
                 'look_back': hp.quniform('look_back', 4, 64, 4)}

    # Trials will contain logging information
        spark_trials = SparkTrials(parallelism=8)
        # Trials=Trials()
    
        with mlflow.start_run():
            BEST = fmin(fn=local_cv, # function to optimize
                    space=SPACE,
                    algo=tpe.suggest, # optimization algorithm Tree Parzen Estimator
                    max_evals=100, # maximum number of iterations
                    trials=spark_trials, # logging
                    rstate=np.random.RandomState(seed=42) # fixing random state for reproducibility
                   )
    
          

        print("Best Local CV {:.4f} params {}".format(local_cv(BEST), BEST))


    # with open("Output_stats.txt", 'a') as out:
    #     out.write("Best performing model chosen hyper-parameters: {}".format(best) + '\n')




#%%




# train_data = tf.data.Dataset.from_tensor_slices((trainX, trainY))
# train_data = train_data.repeat().batch(batch_size, drop_remainder=True)


# # steps_per_epoch = len(trainX) // batch_size 
# for i in range(rep):
#   model.fit(train_data, epochs=30,
#   verbose=2, shuffle=False)
#   model.reset_states()