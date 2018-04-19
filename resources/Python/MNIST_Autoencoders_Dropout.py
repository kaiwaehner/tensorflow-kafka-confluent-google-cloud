
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


# In[4]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt


# In[5]:

print(tf.__version__)
print(np.__version__)
print(matplotlib.__version__)


# In[6]:

mnist = input_data.read_data_sets("mnist_data/")


# In[23]:

tf.reset_default_graph()


# In[24]:

def display_digit(digit):
    plt.imshow(digit.reshape(28, 28), cmap="Greys", interpolation='nearest')


# In[25]:

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


# In[26]:

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs


# In[27]:

X = tf.placeholder(tf.float32, shape=[None, n_inputs])


# In[28]:

dropout_rate = 0.0


# In[29]:

training = tf.placeholder_with_default(False, shape=(), name='training')


# In[30]:

X_drop = tf.layers.dropout(X, dropout_rate, training=training)


# In[31]:

from functools import partial

dense_layer = partial(tf.layers.dense,
                      activation=tf.nn.relu)

hidden1 = dense_layer(X_drop, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3 = dense_layer(hidden2, n_hidden3)

outputs = dense_layer(hidden3, n_outputs, activation=None)


# In[32]:

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))


# In[33]:

optimizer = tf.train.AdamOptimizer(0.01)
training_op = optimizer.minimize(reconstruction_loss)


# In[34]:

init = tf.global_variables_initializer()
saver = tf.train.Saver() 


# In[35]:

n_epochs = 12
batch_size = 100


# In[36]:

with tf.Session() as sess:
    init.run()

    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size

        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)

            sess.run(training_op, feed_dict={X: X_batch, training: True})

        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})   

        print("\r{}".format(epoch), "Train MSE:", loss_train)

        saver.save(sess, "./dropout_autoencoder.ckpt")        


# In[37]:

show_reconstructed_digits(X, outputs, "./dropout_autoencoder.ckpt")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



