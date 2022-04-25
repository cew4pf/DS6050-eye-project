# -*- coding: utf-8 -*-
"""Copy of DL_Proj_Sep_Classes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gr5CCLAjC0kyi3iWReuARXeTbbRx2gcL
"""

# Commented out IPython magic to ensure Python compatibility.
import sys
import sklearn
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import partial
import PIL
import PIL.Image

# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

import pandas as pd
import numpy as np


"""## **Split the data into training and validation sets**"""

# each batch in train_ds or validation_ds tensor: 32 image tensors (224 x 224 x 3); 1D tensor with 32 class labels

from sklearn.datasets import load_files 
from keras.utils import np_utils

from keras.preprocessing import image
from tqdm import tqdm # progress bar

data_dir = "/home/cew4pf/dl_project/content/test_dir/"
batch_size = 32;
# IMPORTANT: Depends on what pre-trained model you choose, you will need to change these dimensions accordingly --> change to 224 since expected size for MobileNet
img_height = 224; 
img_width = 224;

# Training Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 42,
    image_size= (img_height, img_width),
    batch_size = batch_size
)

# Validation Dataset
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 42,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)

validation_ds = validation_ds.prefetch(buffer_size = AUTOTUNE)



DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

model = keras.models.Sequential([
    # rescale images
    tf.keras.layers.Rescaling(scale=1./255.),

    # data augmentation
    tf.keras.layers.RandomFlip('horizontal', seed = 42),
    tf.keras.layers.RandomRotation(0.2, seed = 42),
    #tf.keras.layers.RandomContrast(factor = 0.5, seed = 42),
    tf.keras.layers.RandomTranslation(height_factor = 0.2, width_factor = 0.2, fill_mode = "reflect", seed = 42),
    
    # convolution and max pooling layers
    DefaultConv2D(filters=64, input_shape=[img_height, img_width, 3]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),

    # dense and dropout layers --> add regularizers, increse dropout from 0.2 to 0.5
    keras.layers.Flatten(),
    keras.layers.Dense(units=512, activation='relu'),
    keras.layers.Dense(units=256, activation='relu'),
    keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=64, activation='relu', kernel_regularizer='l1_l2'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=39, activation='softmax'),
])


# class weights

class_weights = {0: 0.11255482156830632,
 1: 0.2283509626922154,
 2: 1.4549517760526935,
 3: 1.5249013806706115,
 4: 1.210608729692699,
 5: 1.6519764957264957,
 6: 1.752372857345233,
 7: 1.332686920922215,
 8: 1.6265614727153188,
 9: 1.6434170320180683,
 10: 1.120775573072393,
 11: 1.2893475088597042,
 12: 1.6961469902646373,
 13: 1.7427444350521275,
 14: 1.7238015607580826,
 15: 1.3383100724872876,
 16: 1.6961469902646373,
 17: 1.6961469902646373,
 18: 1.752372857345233,
 19: 0.4009854452332329,
 20: 1.1450523002869575,
 21: 1.294610151753009,
 22: 1.4616566229469456,
 23: 1.7332212414179629,
 24: 1.5322680540071845,
 25: 1.7427444350521275,
 26: 1.7238015607580826,
 27: 1.7621082621082622,
 28: 1.7238015607580826,
 29: 1.6519764957264957,
 30: 1.0862311204776958,
 31: 1.7427444350521275,
 32: 1.5701954810865701,
 33: 1.7332212414179629,
 34: 1.7621082621082622,
 35: 1.7621082621082622,
 36: 1.7621082621082622,
 37: 1.7621082621082622,
 38: 1.752372857345233}


# model checkpoint callback

checkpoint_filepath = '/home/cew4pf/dl_project/tmp/custom_checkpoint'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# compile model
base_learning_rate = 0.0001

model.compile(loss="sparse_categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(learning_rate = base_learning_rate),
              metrics=["accuracy"])

# fit model
initial_epochs = 400

history = model.fit(train_ds,
                    epochs = initial_epochs,
                    validation_data = validation_ds,
                    callbacks = [model_checkpoint_callback],
                    class_weight = class_weights)

model.load_weights('/home/cew4pf/dl_project/tmp/custom_checkpoint')

model.evaluate(validation_ds)


model.save('/home/cew4pf/dl_project/tmp/custom_model')
