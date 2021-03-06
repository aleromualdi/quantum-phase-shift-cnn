#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

sys.path.append('../')
from potentials import yukawa
from models import cnn

if not os.path.exists('output/'):
    os.makedirs('output/')

import tensorflow as tf
tf.random.set_seed(132)



sources = {
    '0.1': '../../data/yukawa/k=0.1.txt',
    #'0.1_no_bs': '../../data/yukawa/k=0.1Nobounds.txt',
    '0.5': '../../data/yukawa/k=0.5.txt',
    '5.0': '../../data/yukawa/k=5.0.txt',
    '10.0': '../../data/yukawa/k=10.0.txt'
    }


def make_data(k):

    path = sources[k]

    df = pd.read_csv(path, sep="\s+", names=["k", "q", "delta_0", "delta_1", "delta_2"])

    print()
    print('Making potential images for k=%s:'%k)
    print('Data length=', len(df))

    r = np.arange(0.01, 6, 0.06)

    V_vec = []
    for i in range(len(df)):
        q = df.iloc[i]['q']
        v = np.array([yukawa(q, ri, V_0=5) for ri in r])
        V_vec.append(v)

    df['V'] = V_vec

    df.to_pickle('output/data_k%s.pkl'%k)

    return df


def train(k):

    print('processing dataset for k=', k, '...')

    df = make_data(k)

    X = df['V'].values
    y = df['delta_0'].values

    X = np.array([np.array(x) for x in X])
    X = np.expand_dims(X, axis=-1)
    y = np.array(y)
 
    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=3)

    # load the model
    model, model_name = cnn(input_shape=(X.shape[1], 1))

    if not os.path.exists('output/'+model_name):
        os.makedirs('output/'+model_name)

    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor="val_loss",
    #     factor=0.1,
    #     patience=10,
    #     verbose=0,
    #     mode="auto",
    #     min_delta=0.001,
    #     cooldown=0,
    #     min_lr=1e-06,
    # )

    # fit model
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.3,
        verbose=1,
        #callbacks=[reduce_lr]
        )

    model.save('output/'+model_name+'/model_k{}.h5'.format(k))

    with open('output/'+model_name+'/train_hist_k{}.pkl'.format(k), 'wb') as handle:
        pickle.dump(history.history, handle)

    # predict
    predictions = model.predict(X_test)
    predictions = np.array([p[0] for p in predictions]).flatten()
    np.save('output/'+model_name+'/predictions_k{}.npy'.format(k), predictions)
    np.save('output/'+model_name+'/y_test_k%s.npy'%k, y_test)


if __name__ == '__main__':

    for k in sources.keys():
        train(k)

    print('Completed.')
