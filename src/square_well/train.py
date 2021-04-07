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
from models import cnn, cnn_dropout

if not os.path.exists('output/'):
    os.makedirs('output/')


# using TF as descriptor.
sources = {
    #'0.01': '../../data/square_well/k=0.01.txt',
    '0.1': '../../data/square_well/k=0.1.txt',
    }


def make_data(k):

    path = sources[k]

    # load simulation data
    # using TF as descriptor. So a -> q
    df = pd.read_csv(
        path, sep="\s+", names=["k", "q", "delta_0", "delta_0_2", "delta_0_3"])

    # make pontential data
    r = np.linspace(0.1, 8, 100)

    V_vec = []
    for i in range(len(df)):
        q = df.iloc[i]['q']
        # using TF as descriptor.
        scaling_factor = 1.3176
        v = np.array([yukawa(q, ri, V_0=scaling_factor * 0.25) for ri in r])
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
    
    # load the model
    model, model_name = cnn(lr=0.000001, input_shape=(X.shape[1], 1))

    if not os.path.exists('output/'+model_name):
        os.makedirs('output/'+model_name)

    # fit model
    history = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=5000,
        validation_split=0.3,
        verbose=1
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
