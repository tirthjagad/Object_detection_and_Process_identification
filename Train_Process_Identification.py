"""This program is to take the data set and divide them, train and saving the model"""

# Importing the required packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as k
import os
from os import path, makedirs
import sys
import re
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import pandas as pd
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

# Function to load the current working directory


def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
        print(current_path)
    return current_path


# Loding the path of the different files required to the variables
Folder = os.path.join(get_parent_dir(1), "Process_Identification")
Data_Folder = os.path.join(Folder, "Dataset")
Model_Data_Folder = os.path.join(Folder, "Model_Data")
Model_Data = Model_Data_Folder


data_dir = os.path.join(
    Data_Folder,
    "train_timeseries_data.csv")
val_dir = os.path.join(Data_Folder, "val_data.csv")
test_dir = os.path.join(Data_Folder, "test_data.csv")


# Reading the dataframe from the train file
df = pd.read_csv(data_dir)

# Reading the dataframe from the validation file
val_ = pd.read_csv(val_dir)

# Reading the dataframe from the test file
test_df = pd.read_csv(test_dir)

# Reading the first column of the train file
column_indices = {name: i for i, name in enumerate(df.columns)}
n1 = len(val_)
n = len(df)
train_df = df[0:n]
val_df = val_[0:n1]

# Printing the shape of the input to check the size
num_features = df.shape[1]
print(num_features)

# Window generator class which divides the input file to 10 steps to Y_true
# and 11 column as y_pred


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}',
        ])


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack([labels[:, :, self.column_indices[name]]
                           for name in self.label_columns], axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window

# Function for making the data-set which uses keras preprocessing


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)

    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=11,
        shuffle=False,
        batch_size=1)

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


@property
def test(self):
    return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


# Function where the model is created
def _model():
    Model = Sequential([GRU(13, return_sequences=True),  # 1st layer is GRU
                        Flatten(),  # 2layer which merges all 10 time steps to 1
                        Dense(1, activation='relu')])  # 3rd layer fully connected layer

    return Model


Model = _model()

window = WindowGenerator(input_width=10, label_width=1, shift=1, label_columns=[
    'pred_id'])  # label column should be given where the prediction id is there


# Function to train the model
def train_model(model, window):
    # Compiling the model with the hyper-parameters
    model.compile(
        optimizer=Adam(
            learning_rate=0.001), loss=tf.keras.losses.Huber(
            delta=1.5), metrics=['accuracy'])

    # Loading the prarameters for the callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.99,
        patience=5,
        verbose=1)  # parameter for reducing learning rate
    early_stop = EarlyStopping(
        monitor='loss',
        patience=10,
        mode='min',
        verbose=1)  # Parameter to stop model training
    log_dir_time = os.path.join(
        Model_Data,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logging = TensorBoard(log_dir=log_dir_time)

    # Training the model
    history = model.fit(
        window.train,
        epochs=50,
        batch_size=256,
        callbacks=[
            early_stop,
            logging,
            reduce_lr],
        validation_data=window.val,
        verbose=1)

    model.summary()
    model.save(os.path.join(Model_Data, "trained_21_02_2021.h5"))
    return history


train_model(Model, window)
