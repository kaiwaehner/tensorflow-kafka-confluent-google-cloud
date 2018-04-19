
# coding: utf-8

# In[55]:

import pandas as pd 
import numpy as np


# In[56]:

def read_goog_sp500_dataframe():
  """Returns a dataframe with the results for Google and S&P 500"""
  
  # Point to where you've stored the CSV file on your local machine
  googFile = 'data/GOOG.csv'
  spFile = 'data/SP_500.csv'

  goog = pd.read_csv(googFile, sep=",", usecols=[0,5], names=['Date','Goog'], header=0)
  sp = pd.read_csv(spFile, sep=",", usecols=[0,5], names=['Date','SP500'], header=0)

  goog['SP500'] = sp['SP500']

  # The date object is a string, format it as a date
  goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')

  goog = goog.sort_values(['Date'], ascending=[True])

  returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64', 'int64']]]            .pct_change()

  return returns


# In[57]:

def read_goog_sp500_logistic_data():
  """Returns a dataframe with the results for Google and 
  S&P 500 set up for logistic regression"""
  returns = read_goog_sp500_dataframe()

  returns['Intercept'] = 1

  # Leave out the first row since it will not have a prediction for UP/DOWN
  # Leave out the last row as it will not have a value for returns
  # Resultant dataframe with the S&P500 and intercept values of all 1s
  xData = np.array(returns[["SP500", "Intercept"]][1:-1])

  yData = (returns["Goog"] > 0)[1:-1]

  return (xData, yData)


# In[58]:

def read_goog_sp500_data():
  """Returns a tuple with 2 fields, the returns for Google and the S&P 500.
  Each of the returns are in the form of a 1D array"""

  returns = read_goog_sp500_dataframe()

  # Filter out the very first row which does not have any value for returns
  xData = np.array(returns["SP500"])[1:]
  yData = np.array(returns["Goog"])[1:]

  return (xData, yData)


# In[59]:

def read_xom_oil_nasdaq_data():
  """Returns a tuple with 3 fields, the returns for Exxon Mobil, Nasdaq and oil prices.
  Each of the returns are in the form of a 1D array"""

  def readFile(filename):
    # Only read in the date and price at columns 0 and 5
    data = pd.read_csv(filename, sep=",", usecols=[0, 5], names=['Date', 'Price'], header=0)

    # Sort the data in ascending order of date so returns can be calculated
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

    data = data.sort_values(['Date'], ascending=[True])

    # Exclude the date from the percentage change calculation
    returns = data[[key for key in dict(data.dtypes) if dict(data.dtypes)[key] in ['float64', 'int64']]]              .pct_change()

    # Filter out the very first row which has no returns associated with it
    return np.array(returns["Price"])[1:]

  nasdaqData = readFile('data/NASDAQ.csv')
  oilData = readFile('data/USO.csv')
  xomData = readFile('data/XOM.csv')

  return (nasdaqData, oilData, xomData)


# In[60]:

import numpy as np
from sklearn import datasets, linear_model


# In[61]:

nasdaqData, oilData, xomData = read_xom_oil_nasdaq_data()


# In[62]:

combined = np.vstack((nasdaqData , oilData)).T


# In[63]:

xomNasdaqOilModel = linear_model.LinearRegression()


# In[64]:

xomNasdaqOilModel.fit(combined, xomData)
xomNasdaqOilModel.score(combined, xomData)


# In[66]:

print xomNasdaqOilModel.coef_
print xomNasdaqOilModel.intercept_


# In[67]:

import tensorflow as tf


# In[68]:

# Model linear regression y = W1x1 + W2x2 + b
nasdaq_W = tf.Variable(tf.zeros([1, 1]), name="nasdaq_W")
oil_W = tf.Variable(tf.zeros([1, 1]), name="oil_W")


# In[69]:

b = tf.Variable(tf.zeros([1]), name="b")


# In[70]:

nasdaq_x = tf.placeholder(tf.float32, [None, 1], name="nasdaq_x")
oil_x = tf.placeholder(tf.float32, [None, 1], name="oil_x")


# In[71]:

nasdaq_Wx = tf.matmul(nasdaq_x, nasdaq_W)
oil_Wx = tf.matmul(oil_x, oil_W)


# In[72]:

y = nasdaq_Wx + oil_Wx + b


# In[73]:

y_ = tf.placeholder(tf.float32, [None, 1])


# In[74]:

cost = tf.reduce_mean(tf.square(y_ - y))


# In[75]:

train_step_ftrl = tf.train.FtrlOptimizer(1).minimize(cost)


# In[76]:

all_x_nasdaq = nasdaqData.reshape(-1, 1)
all_x_oil = oilData.reshape(-1, 1)
all_ys = xomData.reshape(-1, 1)


# In[77]:

dataset_size = len(oilData)


# In[82]:

def trainWithMultiplePointsPerEpoch(steps, train_step, batch_size):

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)

    for i in range(steps):

      if dataset_size == batch_size:
        batch_start_idx = 0
      elif dataset_size < batch_size:
        raise ValueError("dataset_size: %d, must be greater than batch_size: %d" % (dataset_size, batch_size))
      else:
        batch_start_idx = (i * batch_size) % dataset_size

      batch_end_idx = batch_start_idx + batch_size

      batch_x_nasdaq = all_x_nasdaq[batch_start_idx : batch_end_idx]
      batch_x_oil = all_x_oil[batch_start_idx : batch_end_idx]
      batch_ys = all_ys[batch_start_idx : batch_end_idx]

      feed = { nasdaq_x: batch_x_nasdaq, oil_x: batch_x_oil, y_: batch_ys }

      sess.run(train_step_ftrl, feed_dict=feed)

      # Print result to screen for every 500 iterations
      if (i + 1) % 500 == 0:
        print("After %d iteration:" % i)
        print("W1: %s" % sess.run(nasdaq_W))
        print("W2: %s" % sess.run(oil_W))
        print("b: %f" % sess.run(b))

        print("cost: %f" % sess.run(cost, feed_dict=feed))


# In[83]:

trainWithMultiplePointsPerEpoch(5000, train_step_ftrl, len(oilData))


# In[ ]:



