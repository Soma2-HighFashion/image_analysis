# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import sys
sys.path.append('./data')
import data

import numpy as np
import tflearn as tl
import tflearn.data_utils as du

# Data loading
dataset = data.load_analysis_dataset()

train_X = dataset['data']
train_y = dataset['label']

print("Train Data : ", train_X.shape, train_y.shape)

# Building Residual Network
net = tl.input_data(shape=[None, 42, 42, 3])
net = tl.conv_2d(net, 32, 3)
net = tl.batch_normalization(net)
net = tl.activation(net, 'relu')
net = tl.shallow_residual_block(net, 4, 32, regularizer='L2')
net = tl.shallow_residual_block(net, 1, 32, downsample=True,
											 regularizer='L2')
net = tl.shallow_residual_block(net, 4, 64, regularizer='L2')
net = tl.shallow_residual_block(net, 1, 64, downsample=True,
											 regularizer='L2')
net = tl.shallow_residual_block(net, 5, 64, regularizer='L2')
net = tl.global_avg_pool(net)
# Regression
net = tl.fully_connected(net, 9, activation='softmax')
mom = tl.Momentum(0.1, lr_decay=0.1, decay_step=16000, staircase=True)
net = tl.regression(net, optimizer=mom,
								 loss='categorical_crossentropy')
# Training
model = tl.DNN(net, checkpoint_path='resnet_analysis',
				max_checkpoints=3, tensorboard_verbose=1,
				clip_gradients=1.0)

model.fit(train_X, train_y, n_epoch=200, snapshot_step=500, 
		show_metric=True, batch_size=128, 
		shuffle=True, run_id='resnet_analysis')
