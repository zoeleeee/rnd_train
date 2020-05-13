from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
#from mnist import get_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
import os
import json
from utils import load_data, disorder_data_extend, order_data_extend
import numpy as np

conf = sys.argv[-1]
with open(conf) as config_file:
    config = json.load(config_file)

model_dir = config['model_dir']+'_disorder'
# loss_func = config['loss_func']
loss_func = 'xent'
batch_size = config['training_batch_size']
eval_batch_size = config['eval_batch_size']


x_test = np.load('data/mnist_data.npy')[60000:]
y_test = np.load('data/mnist_labels.npy')[60000:]
def custom_loss():
  def loss(y_true, y_pred):
    if loss_func == 'bce':
      _loss = keras.losses.BinaryCrossentropy()
      return _loss(y_true, tf.nn.sigmoid(y_pred))
    elif loss_func == 'xent':
      _loss = keras.losses.SparseCategoricalCrossentropy()
      return _loss(y_true, tf.nn.softmax(y_pred))
  return loss

print(model_dir)
model = keras.models.load_model(model_dir+'.h5', custom_objects={ 'custom_loss': custom_loss(), 'loss':custom_loss() }, compile=False)

output = model.predict(x_test, batch_size=eval_batch_size)
preds= np.argmax(output, axis=-1)
print(np.mean(preds == y_test))