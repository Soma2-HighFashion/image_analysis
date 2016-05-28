# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import sys
sys.path.append('./data')
import data

import numpy as np
import tflearn as tl
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

dataset = data.load_discriminate_dataset()
train_X = dataset['data']
train_y = dataset['label']

test_X = dataset['data']
test_y = dataset['label']

print("Train Data : ", train_X.shape, train_y.shape)
print("Test Data: ", test_X.shape, test_y.shape)

# Building 'Simple ConvNet'
network = input_data(shape=[None, 42, 42, 3], name='input')

network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')

network = regression(network, optimizer='adam',
					 loss='categorical_crossentropy',
					 learning_rate=0.0001, name='target')

# Model
model = tl.DNN(network, checkpoint_path='discriminator_model', max_checkpoints=3, tensorboard_verbose=1)
model.fit({'input': train_X}, {'target': train_y}, n_epoch=50, shuffle=True,
          show_metric=True, batch_size=32, snapshot_step=300,
		  validation_set=({'input': test_X}, {'target': test_y}),
          snapshot_epoch=False, run_id='convnet_discriminator')

