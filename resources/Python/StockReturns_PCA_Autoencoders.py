
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:

import numpy as np
import pandas as pd
import tensorflow as tf


# In[3]:

get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.mlab import PCA


# In[4]:

print(tf.__version__)
print(np.__version__)
print(pd.__version__)
print(matplotlib.__version__)


# In[5]:

prices = pd.read_csv('stocks.csv')


# In[6]:

prices.head()


# In[7]:

prices['Date'] = pd.to_datetime(prices['Date'], infer_datetime_format=True)


# In[8]:

prices = prices.sort_values(['Date'], ascending=[True])


# In[9]:

prices.head()


# In[11]:

prices = prices[['ADBE', 'MDLZ', 'SBUX']]


# In[12]:

prices.head()


# In[13]:

returns = prices[[key for key in dict(prices.dtypes)     if dict(prices.dtypes)[key] in ['float64', 'int64']]].pct_change()


# In[14]:

returns = returns[1:]


# In[15]:

returns.head()


# In[16]:

returns_arr = returns.as_matrix()[:10]


# In[17]:

returns_arr.shape


# In[18]:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[19]:

returns_arr_scaled = scaler.fit_transform(returns_arr)


# In[20]:

returns_arr_scaled


# In[21]:

results = PCA(returns_arr_scaled, standardize=False)


# In[22]:

results.fracs


# In[23]:

results.Y 


# In[24]:

results.Wt


# In[25]:

np.dot(results.Y, results.Wt)


# In[26]:

returns_arr_scaled


# In[27]:

n_inputs = 3
n_hidden = 2  # codings
n_outputs = n_inputs


# In[28]:

learning_rate = 0.01


# In[29]:

tf.reset_default_graph()


# In[30]:

X = tf.placeholder(tf.float32, shape=[None, n_inputs])


# In[31]:

hidden = tf.layers.dense(X, n_hidden)


# In[32]:

outputs = tf.layers.dense(hidden, n_outputs)


# In[33]:

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))


# In[34]:

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)


# In[35]:

init = tf.global_variables_initializer()


# In[36]:

n_iterations = 10000


# In[37]:

with tf.Session() as sess:
    init.run()

    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: returns_arr_scaled})
    
    outputs_val = outputs.eval(feed_dict={X: returns_arr_scaled})
    print(outputs_val)


# In[38]:

np.dot(results.Y[:,[0,1]], results.Wt[[0,1]])


# In[ ]:




# In[ ]:




# In[ ]:



