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

model_dir = config['model_dir']
nb_labels = config['num_labels']
path = config['permutation']
st_lab = config['start_label']
#np.random.seed(st_lab)
# lab_perm = np.random.permutation(np.load('2_label_permutation.npy')[:nb_labels].T)#[st_lab:st_lab+nb_labels].T)
lab_perm = np.load('2_label_permutation.npy')[st_lab:st_lab+nb_labels].T

# Setting up training parameters
tf.set_random_seed(config['random_seed'])
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']
nb_channal = int(path.split('_')[1].split('.')[1])
# loss_func = config['loss_func']
loss_func = 'xent'

imgs = np.load('data/mnist_data.npy').transpose((0,2,3,1))
labels = keras.utils.to_categorical(np.load('data/mnist_labels.npy'), 10)
input_shape = imgs.shape
nb_labels = labels.shape[-1]
# imgs, labels, input_shape, model_dir = two_pixel_perm_sliding(nb_channal, model_dir)
# imgs, labels, input_shape, model_dir = two_pixel_perm(nb_channal, model_dir)
# imgs, labels, input_shape, model_dir = diff_perm_per_classifier(st_lab, nb_channal, model_dir)
# imgs, labels, input_shape = load_data(path, nb_labels)
print(input_shape)
if loss_func == 'bce':
  labels = np.array([lab_perm[i] for i in labels]).astype(np.float32)

model = keras.Sequential([keras.layers.Conv2D(32, kernel_size=(5,5), padding='same', activation='relu', input_shape=input_shape[1:]),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Conv2D(64, kernel_size=(5,5), activation='relu', padding='same'),
	keras.layers.MaxPooling2D(pool_size=(2,2)),
	keras.layers.Flatten(),
	keras.layers.Dense(1024, activation='relu'),
	keras.layers.Dense(nb_labels)
	])

def custom_loss(y_true, y_pred):
	if loss_func == 'bce':
		loss = keras.losses.BinaryCrossentropy()
		return loss(y_true, tf.nn.sigmoid(y_pred))
	elif loss_func == 'xent':
		loss = keras.losses.SparseCategoricalCrossentropy()
		return loss(y_true, keras.activations.softmax(y_pred))
model.compile(loss=custom_loss, optimizer=keras.optimizers.Adam(1e-3))

x_train, y_train = imgs[:60000], labels[:60000]
x_test, y_test = imgs[60000:], labels[60000:]
N = 10
# x_train, model_dir = disorder_data_extend(N, x_train, model_dir)
x_train, model_dir = order_data_extend(N, x_train, model_dir)
y_train = np.vstack(list(y_train)*N)

epochs = max_num_training_steps * batch_size / len(x_train)

#<<<<<<< HEAD
#model.fit(x_train, y_train, batch_size=batch_size, epochs=int(epochs), verbose=2, validation_data=(x_test,y_test))
#=======
chkpt_cb = tf.keras.callbacks.ModelCheckpoint(model_dir+'.h5',
                                              monitor='val_loss',
                                              save_best_only=True,
                                              mode='min')

model.fit(x_train, y_train, batch_size=batch_size, epochs=int(epochs), verbose=2, validation_data=(x_test,y_test), callbacks=[chkpt_cb])
#>>>>>>> refs/remotes/origin/master

#model.save(model_dir+'.h5')

