
# coding: utf-8

# In[16]:

import numpy as np
import tensorflow as tf
import google.datalab.ml as ml


# In[17]:

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


# In[18]:

# Store the MNIST data in a folder
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)


# In[19]:

training_digits, training_labels = mnist.train.next_batch(5000)
test_digits, test_labels = mnist.test.next_batch(200)


# In[20]:

training_digits_pl = tf.placeholder("float", [None, 784])


# In[21]:

test_digit_pl = tf.placeholder("float", [784])


# In[22]:

# Nearest Neighbor calculation using L1 distance
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digit_pl)))


# In[23]:

distance = tf.reduce_sum(l1_distance, axis=1)


# In[24]:

# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.


# In[25]:

# Initializing the variables
init = tf.global_variables_initializer()


# In[26]:

with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(test_digits)):
        # Get nearest neighbor
        nn_index = sess.run(pred,             feed_dict={training_digits_pl: training_digits, test_digit_pl: test_digits[i, :]})

        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(training_labels[nn_index]),             "True Label:", np.argmax(test_labels[i]))

        # Calculate accuracy
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1./len(test_digits)

    print("Done!")
    print("Accuracy:", accuracy)


# In[ ]:



