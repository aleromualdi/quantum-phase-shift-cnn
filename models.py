#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

rmse = tf.keras.metrics.RootMeanSquaredError()


def cnn(input_shape, loss='mse', metrics=['mse', rmse, 'mae'],  name='cnn' ):
    '''
    CNN model based on convolution layer + max pooling layer
    
    Note:
    paddng=same results in padding evenly to the left/right or up/down of
        the input such that output has the same height/width dimension as the input
  
    '''

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(32, 2, strides=2, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, 2, strides=2, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, 2, strides=2, padding='same', activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(256, 2, strides=1, padding='same', activation='relu'))
    # model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(units=3))

    opt =tf.keras.optimizers.Adam(lr=1e-04)

    model.compile(
                optimizer=opt,
                loss=loss,
                metrics=metrics
                )
    
    return model, name


def cnn3(input_shape, loss='mse', metrics=['mse', rmse, 'mae'], model_name="cnn3"):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),



        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1)])
    
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-04),
            loss=loss,
            metrics=metrics
    )

    return model, model_name


def cnn4(input_shape, loss='mse', metrics=['mse', rmse, 'mae'], model_name="cnn3"):
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv1D(16, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        #tf.keras.layers.Dropout(0.1),
    
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
    
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.15),

        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1)
        ])
    
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-04),
            loss=loss,
            metrics=metrics
    )

    return model, model_name

def dense(input_shape, loss='mae', metrics=['mse', rmse, 'mae'], model_name="dense"):
    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-04),
            loss=loss,
            metrics=metrics
    )
    return model, model_name
