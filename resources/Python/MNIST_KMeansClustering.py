
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf

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


# In[6]:

training_digits, training_labels = mnist.train.next_batch(2000)

test_digits, test_labels = mnist.test.next_batch(5)


# In[7]:

def display_digit(digit):
    plt.imshow(digit.reshape(28, 28), cmap="Greys", interpolation='nearest')


# In[8]:

display_digit(training_digits[0])


# In[9]:

from tensorflow.contrib.learn.python.learn.estimators import kmeans

from tensorflow.contrib.factorization.python.ops import clustering_ops


# In[10]:

def input_fn(digits):
    input_t = tf.convert_to_tensor(digits, dtype=tf.float32)
    
    return (input_t, None)


# In[11]:

k_means_estimator = kmeans.KMeansClustering(num_clusters=10)


# In[12]:

fit = k_means_estimator.fit(input_fn=lambda: input_fn(training_digits), steps=1000)


# In[13]:

clusters = k_means_estimator.clusters()


# In[14]:

for i in range(10):
    plt.subplot(2, 5, i + 1)
    display_digit(clusters[i])


# In[15]:

cluster_labels = [9, 1, 6, 8, 0, 3, 0, 6, 1, 9]


# In[16]:

for i in range(5):
    plt.subplot(1, 5, i + 1)
    display_digit(test_digits[i])


# In[17]:

predict = k_means_estimator.predict(input_fn=lambda: input_fn(test_digits), as_iterable=False)


# In[18]:

predict


# In[19]:

print([cluster_labels[i] for i in predict['cluster_idx']])


# In[20]:

for i in range(5):
    plt.subplot(1, 5, i + 1)
    display_digit(test_digits[i])


# In[21]:

training_labels[:5]


# In[22]:

predict_train = k_means_estimator.predict(input_fn=lambda: input_fn(test_digits), as_iterable=False)


# In[23]:

def display_accuracy(cluster_labels, cluster_idx, actual_labels):

    predict_labels = [cluster_labels[i] for i in cluster_idx]
    
    num_accurate_predictions = (list(predict_labels == actual_labels)).count(True)
    
    print("Number of accurate predictions: ", num_accurate_predictions)

    pctAccuracy = float(num_accurate_predictions) / float(len(actual_labels))

    print("% accurate predictions: ", pctAccuracy)    


# In[24]:

display_accuracy(cluster_labels, predict_train['cluster_idx'], test_labels)


# In[ ]:




# In[ ]:



