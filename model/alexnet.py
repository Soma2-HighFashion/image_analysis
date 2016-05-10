# coding: utf-8

import tensorflow as tf
import numpy as np

class AlexNet:

	def __init__(self, geometry, num_classes):
		self._init_weight(geometry, num_classes)

	def _init_weight(self, geometry, num_classes):
		# Tensor Flow Graph Input
		self.X = tf.placeholder(tf.float32, [None, geometry[0]*geometry[1]])
		self.y = tf.placeholder(tf.float32, [None, num_classes])
		self.dropout = tf.placeholder(tf.float32)

		self.weight = {
			'Wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
			'Wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
			'Wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
			'Wf1': tf.Variable(tf.random_normal([4*4*256, 1024])),
			'Wf2': tf.Variable(tf.random_normal([1024, 1024])),
			'Wout': tf.Variable(tf.random_normal([1024, num_classes]))
		}

		self.bias = {
			'bc1': tf.Variable(tf.random_normal([64])),
			'bc2': tf.Variable(tf.random_normal([128])),
			'bc3': tf.Variable(tf.random_normal([256])),
			'bf1': tf.Variable(tf.random_normal([1024])),
			'bf2': tf.Variable(tf.random_normal([1024])),
			'bout': tf.Variable(tf.random_normal([num_classes]))
		}

	def _model(self):
		# Convolution Network

		# Reshape input picture
		input_X = tf.reshape(self.X, shape=[-1, 128, 128, 3])

		# Stage 1 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
		conv1 = tf.nn.conv2d(input_X, self.weight['Wc1'], strides = [1, 1, 1, 1], padding='SAME')
		conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.bias['bc1']))
		conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
		conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
		conv1 = tf.nn.dropout(conv1, self.dropout)

		# Stage 2 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
		conv2 = tf.nn.conv2d(conv1, self.weight['Wc2'], strides = [1, 1, 1, 1], padding='SAME')
		conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.bias['bc2']))
		conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
		conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
		conv2 = tf.nn.dropout(conv2, self.dropout)

		# Stage 3 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
		conv3 = tf.nn.conv2d(conv2, self.weight['Wc3'], strides = [1, 1, 1, 1], padding='SAME')
		conv3 = tf.nn.relu(tf.nn.bias_add(conv3, self.bias['bc3']))
		conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
		conv3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
		conv3 = tf.nn.dropout(conv3, self.dropout)

		# Stage 4 : Fully connected : Linear -> ReLU -> Linear
		fc1 = tf.reshape(conv3, [-1, self.weight['Wf1'].get_shape().as_list()[0]])
		fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, self.weight['Wf1']), self.bias['bf1']))
		fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, self.weight['Wf2']), self.bias['bf2']))

		out = tf.matmul(fc2, self.weight['Wout']) + self.bias['bout']
		return out

	def train(self, mnist, learning_rate = 0.001, num_iters = 1000, batch_size=50, dropout_prob=0.5, verbose=False):
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

			writer = tf.train.SummaryWriter("/tmp/tb_test", sess.graph_def)
			
			# Train
			for epoch in range(1, num_iters+1):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				# Fit training data
				sess.run(optimizer, feed_dict={self.X: batch_xs, self.y: batch_ys, self.dropout: dropout_prob})
	
				merged_summ = sess.run(merged, feed_dict={self.X: batch_xs, self.y: batch_ys, self.dropout: dropout_prob})
				writer.add_summary(merged_summ, epoch)

				if epoch & 50 == 0:
					loss = sess.run(cost, feed_dict={self.X: batch_xs, self.y: batch_ys, self.dropout: 1.})
					acc = sess.run(accuracy, feed_dict={self.X: batch_xs, self.y: batch_ys, self.dropout: 1.})
					print "Epoch : ", epoch, " loss=" , loss, " Trainig Accuracy=", acc
			
			print "Optimization Finishied"
			saver.save(sess, "./tmp/alexnet_model.ckpt")
			
	def predict(self, X, y):
		# Evaluate model
		pred_y = self._model()
		correct_pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(self.y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, "./tmp/alexnet_model.ckpt")
			print sess.run(accuracy, feed_dict={self.X: X, self.y: y, self.dropout: 1. })
