# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 19:42:51 2019

@author: Eric Born
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM,\
                                    BatchNormalization, Flatten,\
                                    Embedding, Bidirectional, Attention,\
                                    TimeDistributed
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint,\
                                              TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import RMSprop
from keras.backend.tensorflow_backend import set_session 
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
import time

# removes scientific notation from np prints, prints numbers as floats
np.set_printoptions(suppress=True)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# path to data
path = 'G:/botdata/win/'

# list of all files
files = os.listdir(path)

# object for single array file, which equals a single game
#full_array = np.load(path + files[0], allow_pickle=True)

# column list for all match data
cols = ['minerals','gas','supply_cap', 'supply_army', 'supply_workers',
        'nexus', 'c_pylons', 'assimilators', 'gateways', 'cybercore', 'robofac',
        'stargate', 'robobay', 'k-structures', 'k-units', 'attack',
        'assimilators', 'offensive_force', 'b_pylons', 'workers', 'distribute',
        'nothing', 'expand', 'buildings', 'ZEALOT', 'STALKER', 'ADEPT',
        'IMMORTAL', 'VOIDRAY', 'COLOSSUS', 'difficulty', 'outcome']
#
# Single array df
#full_df = pd.DataFrame(data=full_array,columns=cols)

# define empty dataframe using columns previously defined
full_df = pd.DataFrame(columns=cols)

# loads each training data file, creates a df and appends to the original df
for file in range(0, len(files)):
    #print(np.load(path + files[file], allow_pickle=True))
    if len(full_df) == 0:
        full_df = pd.DataFrame(data=np.load(path + files[file], 
                                            allow_pickle=True),columns=cols)
    else:
        df2 = pd.DataFrame(data=np.load(path + files[file], 
                                allow_pickle=True),columns=cols)
        full_df = full_df.append(df2)

# setup x and y
# x will be the supply_data inputs
# y will be the action that was taken

# create values, the supply stats
x_data = full_df.iloc[:,0:15].values

# create targets, the bot choices
y_data = full_df.iloc[:,15:24].values

# used for a single array load
#supply_array = full_array[:,0:15]
#x_data = full_array[:,0:15]

# setup an array that only includes the action data index 15-23
# 9 outputs
#actions_array = full_array[:,15:24]
#y_data = full_array[:,15:24]

#np.save(r"C:/botdata/{}.npy".format(str(int(time.time()))),
#        np.array(self.training_data))

# look at the shape of the array
print(x_data.shape)
print(y_data.shape)

# setup number to split for train/test
num_train = int(0.9 * len(x_data))

# split x train/test
x_train = x_data[0:num_train]
x_test = x_data[num_train:]
len(x_train) + len(x_test)

# split y train/test
y_train = y_data[0:num_train]
y_test = y_data[num_train:]
len(y_train) + len(y_test)

# input columns
num_x_signals = x_data.shape[1]
print(num_x_signals)

# output columns
num_y_signals = y_data.shape[1]
print(num_y_signals)

# Setup scalers for x and y
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.fit_transform(x_test)

y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.fit_transform(y_test)

# check shapes for the scaled sets
print(x_train_scaled.shape)
print(y_train_scaled.shape)

# create a batch generator to feed the data into the NN
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx+sequence_length]
            y_batch[i] = y_train_scaled[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)

# larger batch size the more is fed to the GPU at once, adjust if there
# are memory issues
batch_size = 200

# length of steps
sequence_length = 300

# create a generator object
generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)

# generates a batch for x and y
x_batch, y_batch = next(generator)

#print(x_batch.shape)
#print(y_batch.shape)

# set aside some data for validation purposes
validation_data = (np.expand_dims(x_test_scaled[0:500], axis=0),
                   np.expand_dims(y_test_scaled[0:500], axis=0))

# start building the NN model
# Sequential allows the model to be build one layer at a time
# with each subsequent layer being added to the first
model = Sequential()

# trying out various different model constructions
#model.add(CuDNNLSTM(units=64, return_sequences=True, 
#               input_shape=(None, num_x_signals,)))

#model.add(Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.4, 
#                             activation='relu', return_sequences=True)))

#model.add(Bidirectional(CuDNNLSTM(32, return_sequences = True)))

#model.add(Bidirectional(LSTM(64, dropout=0.4, recurrent_dropout=0.4, 
#                             activation='relu', return_sequences=True)))

#model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))

# The GRU outputs a batch of sequences of 512 values. 
# We want to predict 9 output-signals, so we add a fully-connected (or dense) 
# layer maps 512 values down to only 9 values.
#model.add(TimeDistributed(Dense(num_y_signals, activation='sigmoid')))

# 60.5%
model.add(CuDNNLSTM(units=60, return_sequences=True, 
               input_shape=(None, num_x_signals,)))

model.add(Bidirectional(CuDNNLSTM(units=15, return_sequences=True)))

model.add(TimeDistributed(Dense(num_y_signals, activation='sigmoid')))


# 57%
#model.add(Bidirectional(CuDNNLSTM(units=20, return_sequences=True), 
#               input_shape=(None, num_x_signals,)))

#model.add(TimeDistributed(Dense(num_y_signals, activation='sigmoid')))

if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))

warmup_steps = 50

def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.
    
    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean

optimizer = RMSprop(lr=1e-3)
# original loss and optimizer
#model.compile(loss=loss_mse_warmup, optimizer=optimizer)

# suggested for LSTM
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics=['categorical_accuracy'])
print(model.summary())

# build writing checkpoints
path_checkpoint = 'CuDNNLSTM_expanded_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)

# set early stop if the performance worsens on validation
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)

# setup tensorboard
callback_tensorboard = TensorBoard(
    log_dir=r'C:/Users/TomBrody/Desktop/School/767 ML/SC Bot/NN/logs/CuDNNLSTM',
                                   histogram_freq=0,
                                   write_graph=False)

# setup automatic reduction of learning rate
# TODO
#try with patience at 5
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-8,
                                       patience=5,
                                       verbose=1)

# setup list to hold all callback info
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

# Run the model for 20 epochs, 100 steps per
model.fit_generator(generator=generator,
                    epochs=100,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks)

# load best model
try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

# path to save the model
path = r'C:/Users/TomBrody/Desktop/School/767 ML/SC Bot/NN/model//'

# path plus model and time
model.save(path+'CuDNNLSTM-{}'.format(str(int(time.time())))+'.h5')
