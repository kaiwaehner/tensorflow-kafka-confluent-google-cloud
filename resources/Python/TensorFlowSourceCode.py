
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


# In[3]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt


# In[4]:

print(tf.__version__)
print(np.__version__)
print(matplotlib.__version__)


# In[6]:

import random

input_2d_x_1 = np.array([[random.randint(1, 1000) for i in range(2)] for j in range(100)], dtype=np.float32)
input_2d_x_2 = np.array([[random.randint(700, 2000) for i in range(2)] for j in range(100)], dtype=np.float32)
input_2d_x_3 = np.array([[random.randint(1700, 3000) for i in range(2)] for j in range(100)], dtype=np.float32)

input_2d_x = np.append(np.append(input_2d_x_1, input_2d_x_2, axis=0), input_2d_x_3, axis=0)


# In[7]:

input_2d_x


# In[8]:

def input_fn_2d(input_2d):
    input_t = tf.convert_to_tensor(input_2d, dtype=tf.float32)
    
    return (input_t, None)


# In[9]:

plt.scatter(input_2d_x[:,0], input_2d_x[:,1], s=100, color="green")
plt.show()


# In[10]:

from tensorflow.contrib.learn.python.learn.estimators import kmeans

from tensorflow.contrib.factorization.python.ops import clustering_ops


# In[11]:

k_means_estimator = kmeans.KMeansClustering(num_clusters=3)


# In[12]:

fit = k_means_estimator.fit(input_fn=lambda: input_fn_2d(input_2d_x), steps=1000)


# In[13]:

clusters_2d = k_means_estimator.clusters()
clusters_2d


# In[14]:

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(input_2d_x[:,0], input_2d_x[:,1], s=100, marker='o', color="green")
ax1.scatter(clusters_2d[:,0], clusters_2d[:,1], c='r', s=300, marker='s')

plt.show()


# In[15]:

k_means_estimator.get_params()


# In[16]:

ex_2d_x = np.array([[1700, 1700]], dtype=np.float32)


# In[17]:

predict = k_means_estimator.predict(input_fn=lambda: input_fn_2d(ex_2d_x), as_iterable=False)


# In[18]:

predict


# In[ ]:




# In[ ]:



