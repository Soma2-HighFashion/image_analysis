# -*- coding: utf-8 -*-

""" Very Deep Convolutional Networks for Large-Scale Visual Recognition.
Applying VGG 16-layers convolutional network to Oxford's 17 Category Flower
Dataset classification task.
References:
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    K. Simonyan, A. Zisserman. arXiv technical report, 2014.
Links:
    http://arxiv.org/pdf/1409.1556
"""

from __future__ import division, print_function, absolute_import

import sys
sys.path.append('./data')
from data import generate_patches, img2numpy_arr

import argparse
import numpy as np
import tflearn as tl
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-test_path', action='store', dest='test_path', type=str, 
			help='Test Data Path')	
	config = parser.parse_args()

	# Load Test Data
	X = generate_patches(img2numpy_arr(config.test_path))

	# Building 'VGG Network'
	network = input_data(shape=[None, 84, 84, 3])

	network = conv_2d(network, 64, 3, activation='relu')
	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 128, 3, activation='relu')
	network = conv_2d(network, 128, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = conv_2d(network, 256, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = conv_2d(network, 512, 3, activation='relu')
	network = max_pool_2d(network, 2, strides=2)

	network = fully_connected(network, 4096, activation='relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 4096, activation='relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 2, activation='softmax')

	network = regression(network, optimizer='rmsprop',
						 loss='categorical_crossentropy',
						 learning_rate=0.00001)

	# Model
	model = tl.DNN(network)

	model.load('vgg_gender_model-80000')
	pred_y = model.predict(X)
	print(pred_y)
