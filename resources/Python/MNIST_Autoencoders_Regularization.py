
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:

import numpy as np
import tensorflow as tf

import sys

from tensorflow.examples.tutorials.mnist import input_data


# In[3]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt


# In[4]:

print(tf.__version__)
print(np.__version__)
print(matplotlib.__version__)


# In[5]:

mnist = input_data.read_data_sets("mnist_data/")


# In[25]:

tf.reset_default_graph()


# In[26]:

def display_digit(digit):
    plt.imshow(digit.reshape(28, 28), cmap="Greys", interpolation='nearest')


# In[27]:

def show_reconstructed_digits(X, outputs, model_path = None):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[100 : 102]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 6))
    for i in range(2):
        plt.subplot(2, 2, i * 2 + 1)
        display_digit(X_test[i])
        plt.subplot(2, 2, i * 2 + 2)
        display_digit(outputs_val[i])


# In[28]:

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150 
n_hidden3 = n_hidden1
n_outputs = n_inputs


# In[29]:

he_init = tf.contrib.layers.variance_scaling_initializer()


# In[30]:

l2_regularizer = tf.contrib.layers.l2_regularizer(0.0001)


# In[31]:

X = tf.placeholder(tf.float32, shape=[None, n_inputs])


# In[32]:

from functools import partial

dense_layer = partial(tf.layers.dense,
                      activation=tf.nn.elu)


# In[33]:

hidden1 = dense_layer(X, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3 = dense_layer(hidden2, n_hidden3)

outputs = dense_layer(hidden3, n_outputs, activation=None)


# In[34]:

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))


# In[35]:

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)


# In[36]:

loss = tf.add_n([reconstruction_loss] + reg_losses)


# In[37]:

optimizer = tf.train.AdamOptimizer(0.01)
training_op = optimizer.minimize(loss)


# In[38]:

init = tf.global_variables_initializer()


# In[39]:

saver = tf.train.Saver()


# In[40]:

n_epochs = 8
batch_size = 150


# In[41]:

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size

        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})

        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})   
        print("\r{}".format(epoch), "Train MSE:", loss_train)

        saver.save(sess, "./regularization_autoencoder.ckpt")
        


# In[42]:

show_reconstructed_digits(X, outputs, "./regularization_autoencoder.ckpt")


# In[ ]:



