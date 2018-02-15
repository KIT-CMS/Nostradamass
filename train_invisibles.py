from os import environ
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ['CUDA_VISIBLE_DEVICES'] = "1"
import copy
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import math
import numpy as np
seed = 1234
np.random.seed(seed)

from matplotlib.colors import LogNorm

selected_channel = 'tt'

from common_functions import add_pu_target
from common_functions import transform_fourvector
from common_functions import load_from_log, load_from_pickle, load_model
from losses import custom_loss

def train_model(X, Y, model_filename = "toy_mass.h5", out_folder='', previous_model=None):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    from keras.layers import GaussianNoise
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    set_session(sess)
    kernel_initializer = "random_uniform"
    bias_initializer = "Zeros"
    X, Y = add_pu_target(X, Y, 6., 0.0, 24.)
    
    if previous_model == None:    
        model = Sequential()
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, input_shape=(X.shape[1],)))
        model.add(GaussianNoise(stddev=1.0))
#        model.add(Dropout(0.05))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dropout(0.05))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dropout(0.05))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
#        model.add(Dropout(0.05))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(300, activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer))
        model.add(Dense(Y.shape[1], activation='linear'))
        model.compile(loss=custom_loss, optimizer='adam')
    else:
        model = load_model(previous_model)

    model.summary()
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(patience = 50)

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    tmp_X = X_train 

    for i in range(1):
        model_checkpoint = ModelCheckpoint( os.path.join(out_folder, 'model.'+str(i)+'.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                            monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1)
        model.fit(tmp_X, Y_train, # Training data
                    batch_size=50000, # Batch size
                    epochs=1000, # Number of training epochs
                    validation_data = (X_test, Y_test),
                    callbacks = [model_checkpoint, early_stopping])
    model.save(os.path.join(out_folder, model_filename))
    return model
    
if __name__ == '__main__':
    in_filename = sys.argv[1]
    out_folder = sys.argv[2]
    if len(sys.argv) > 3:
        previous_model = sys.argv[3]
    else:
        previous_model = None
    print "previous: ", previous_model
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if in_filename[-4:] == ".log":
        X, Y, B, M, L, phys_M = load_from_log(in_filename, "pickle.pkl", out_folder=out_folder, save_cache=True)
    elif in_filename[-4:] == ".pkl":
        X, Y, B, M, L, phys_M = load_from_pickle(in_filename)

    model = train_model(X, Y, out_folder=out_folder, previous_model = previous_model)
    model.summary()
