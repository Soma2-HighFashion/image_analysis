
# coding: utf-8

# ## Reference
# - [TensorFlow Example - convolution network](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3%20-%20Neural%20Networks/convolutional_network.ipynb)

# In[2]:

import tensorflow as tf
import numpy as np
import input_data

mnist = input_data.read_data_sets("./", one_hot=True)


# In[3]:

# Parameters
learning_rate = 0.0001
num_iters = 1500
batch_size = 64

geometry = [28, 28]
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_classes = len(classes)
dropout_prob = 0.8


# In[4]:

# Tensor Flow Graph Input
X = tf.placeholder(tf.float32, [None, geometry[0]*geometry[1]])
y = tf.placeholder(tf.float32, [None, num_classes])
dropout = tf.placeholder(tf.float32)

# AlexNet Weight & bias
# 3x3 conv, 1 input, 64 outputs
Wc1 = tf.Variable(tf.random_normal([3, 3, 1, 64]))
bc1 = tf.Variable(tf.random_normal([64]))

# 3x3 conv, 64 input, 128 outputs
Wc2 = tf.Variable(tf.random_normal([3, 3, 64, 128]))
bc2 = tf.Variable(tf.random_normal([128]))

# 3x3 conv, 128 input, 256 outputs
Wc3 = tf.Variable(tf.random_normal([3, 3, 128, 256]))
bc3 = tf.Variable(tf.random_normal([256]))

# Fully connected (Standard 3-layer MLP), 4*4*256 input, 1024 
Wf1 = tf.Variable(tf.random_normal([4*4*256, 1024]))
bf1 = tf.Variable(tf.random_normal([1024]))

Wf2 = tf.Variable(tf.random_normal([1024, 1024]))
bf2 = tf.Variable(tf.random_normal([1024]))

Wout = tf.Variable(tf.random_normal([1024, num_classes]))
bout = tf.Variable(tf.random_normal([num_classes]))


# In[5]:

# Convolution Network

# Reshape input picture
input_X = tf.reshape(X, shape=[-1, 28, 28, 1])

# Stage 1 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
conv1 = tf.nn.conv2d(input_X, Wc1, strides = [1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.relu(tf.nn.bias_add(conv1, bc1))
conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
conv1 = tf.nn.dropout(conv1, dropout)

# Stage 2 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
conv2 = tf.nn.conv2d(conv1, Wc2, strides = [1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.relu(tf.nn.bias_add(conv2, bc2))
conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
conv2 = tf.nn.dropout(conv2, dropout)

# Stage 3 : Convolution -> ReLU -> Max Pooling -> Local Response Normalization -> Dropout
conv3 = tf.nn.conv2d(conv2, Wc3, strides = [1, 1, 1, 1], padding='SAME')
conv3 = tf.nn.relu(tf.nn.bias_add(conv3, bc3))
conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
conv3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
conv3 = tf.nn.dropout(conv3, dropout)

# Stage 4 : Fully connected : Linear -> ReLU -> Linear
fc1 = tf.reshape(conv3, [-1, Wf1.get_shape().as_list()[0]])
fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, Wf1), bf1))
fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, Wf2), bf2))

out = tf.add(tf.matmul(fc2, Wout), bout)


# In[6]:

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# In[ ]:

init = tf.initialize_all_variables()

# Launch the Graph
with tf.Session() as sess:
    sess.run(init)
    
    # Train
    for epoch in range(1, num_iters+1):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training data
        
        sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout: dropout_prob})
        
        if epoch & 50 == 0:
			loss = sess.run(cost, feed_dict={X: batch_xs, y: batch_ys, dropout: 1.})
			acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, dropout: 1.})
			print "Epoch : ", epoch, " loss=" , loss, " Training Accuracy=", acc
    
    print("Optimization Finishied")
    
    # Test
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, 
                                                             y: mnist.test.labels, 
                                                             dropout: 1.}) )


# In[ ]:




# In[ ]:



