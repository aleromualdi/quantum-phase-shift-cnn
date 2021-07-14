
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# load model trained on double-Yukawa 
dset_name = 'k0.1'
model = load_model('output/cnn/model_%s.h5'%dset_name)

# load Yukawa data
#df = pd.read_pickle('../yukawa/output/data_%s.pkl'%dset_name)
df = pd.read_pickle('../yukawa/output/data_k0.1_no_bs.pkl')

X = df['V'].values
y = df['delta_0'].values

X = np.array([np.array(x) for x in X])
X = np.expand_dims(X, axis=-1)

print('predicting...')
predictions = model.predict(X).flatten()
np.save('output/predict_yukawa/predictions_%s.npy'%dset_name, predictions)
np.save('output/predict_yukawa/y_test_%s.npy'%dset_name, y)
print('Done.')


