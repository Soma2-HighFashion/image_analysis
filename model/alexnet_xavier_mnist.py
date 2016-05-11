# coding: utf-8

import time
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d
import numpy as np

class AlexNet:

	def __init__(self, geometry, num_classes):
		self.geometry = geometry
		self.num_classes = num_classes
		
		# Tensor Flow Graph Input
		self.X = tf.placeholder(tf.float32, [None, self.geometry[0] * self.geometry[1]])
		self.y = tf.placeholder(tf.float32, [None, self.num_classes])
		self.dropout = tf.placeholder(tf.float32)

	def _model(self):
		# Reshape input picture
		input_X = tf.reshape(self.X, shape=[-1, self.geometry[0], self.geometry[1], 1])
		
		# Stage 1 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
		conv1_name = "c1"
		conv1 = self.__conv_relu(input_X, [3, 3, 1, 64], [64], conv1_name)
		conv1 = self.__max_pooling(conv1, k=2, name=conv1_name)
		conv1 = self.__local_response_norm(conv1, name=conv1_name)
		conv1 = self.__dropout(conv1, self.dropout)

		# Stage 2 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
		conv2_name = "c2"
		conv2 = self.__conv_relu(conv1, [3, 3, 64, 128], [128], conv2_name)
		conv2 = self.__max_pooling(conv2, k=2, name=conv2_name)
		conv2 = self.__local_response_norm(conv2, name=conv2_name)
		conv2 = self.__dropout(conv2, self.dropout)
		
		# Stage 3 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
		conv3_name = "c3"
		conv3 = self.__conv_relu(conv2, [3, 3, 128, 256], [256], conv3_name)
		conv3 = self.__max_pooling(conv3, k=2, name=conv3_name)
		conv3 = self.__local_response_norm(conv3, name=conv3_name)
		conv3 = self.__dropout(conv3, self.dropout)

		# Stage 4 : Fully connected : Linear -> ReLU -> Linear
		input_conv_X = tf.reshape(conv3, [-1, 4*4*256])

		fc1_name = "f1"
		fc1 = self.__fc_relu(input_conv_X, [4*4*256, 1024], [1024], fc1_name) 
		
		fc2_name = "f2"
		fc2 = self.__fc_relu(fc1, [1024, 512], [512], fc2_name) 

		out_name = "out"
		out = self.__linear(fc2, [512, self.num_classes], out_name)
		return out

	def __conv_relu(self, input, kernel_shape, bias_shape, name):
		weights = tf.get_variable("W"+name, kernel_shape, initializer=xavier_initializer_conv2d())
		biases = tf.get_variable("b"+name, bias_shape, initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding='SAME', name=name+"_conv")
		return tf.nn.relu(conv + biases)

	def __max_pooling(self, input, k, name):
		return tf.nn.max_pool(input, 
			ksize=[1, k, k, 1], strides = [1, k, k, 1],
			padding='SAME', name=name+"_maxpool")

	def __local_response_norm(self, input, name):
		return tf.nn.lrn(input, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name+"_lrn")

	def __dropout(self, input, dropout_prob):
		return tf.nn.dropout(input, dropout_prob)

	def __linear(self, input, kernel_shape, name):
		weights = tf.get_variable("W"+name, kernel_shape, initializer=xavier_initializer())
		return tf.matmul(input, weights)

	def __fc_relu(self, input, kernel_shape, bias_shape, name):
		biases = tf.get_variable("b"+name, bias_shape, initializer=tf.constant_initializer(0.0))
		return tf.nn.relu(self.__linear(input, kernel_shape, name) + biases)

	def train(self, mnist, learning_rate = 0.001, num_iters = 1000, 
			batch_size=50, dropout_prob=0.5, verbose=False):
		
		# Loss and Optimizer
		pred_y = self._model()
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_y, self.y))
		loss_summ = tf.scalar_summary("cross entropy", cost)

		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		acc_summ = tf.scalar_summary("accuracy", accuracy)

		merged = tf.merge_all_summaries()
		saver = tf.train.Saver()

		init = tf.initialize_all_variables()

		# Launch the Graph
		with tf.Session() as sess:
			sess.run(init)

			writer = tf.train.SummaryWriter("/tmp/tb_test", sess.graph)
			
			# Train
			for epoch in range(1, num_iters+1):
				start_time = time.time()

				batch_xs, batch_ys = mnist.train.next_batch(batch_size)

				# Fit training data
				sess.run(
					optimizer, 
					feed_dict={
						self.X: batch_xs, 
						self.y: batch_ys, 
						self.dropout: dropout_prob
					}
				)

				merged_summ = sess.run(
								merged, 
								feed_dict={
									self.X: batch_xs, 
									self.y: batch_ys, 
									self.dropout: dropout_prob
								}
							)
				writer.add_summary(merged_summ, epoch)

				loss = sess.run(
						cost, 
						feed_dict={
							self.X: batch_xs, 
							self.y: batch_ys, 
							self.dropout: 1.
						}
					)
				acc = sess.run(
						accuracy, 
						feed_dict={
							self.X: batch_xs, 
							self.y: batch_ys, 
							self.dropout: 1.
						}
					)
				print "Epoch ", epoch, "- Time Taken:", (time.time() - start_time)  ,\
					"sec, Loss:" , loss, ", Training Accuracy:", acc
			
			print "Optimization Finishied"
			saver.save(sess, "./tmp/alexnet_model_mnist.ckpt")
			
	def predict(self, X, y):
		# Evaluate model
		pred_y = self._model()
		correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, "./tmp/alexnet_model_mnist.ckpt")
			print sess.run(accuracy, feed_dict={self.X: X, self.y: y, self.dropout: 1. })
