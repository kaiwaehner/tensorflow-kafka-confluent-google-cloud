
# coding: utf-8

# In[71]:

import pandas as pd 
import numpy as np


# In[72]:

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


# In[73]:

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


# In[74]:

def read_goog_sp500_data():
  """Returns a tuple with 2 fields, the returns for Google and the S&P 500.
  Each of the returns are in the form of a 1D array"""

  returns = read_goog_sp500_dataframe()

  # Filter out the very first row which does not have any value for returns
  xData = np.array(returns["SP500"])[1:]
  yData = np.array(returns["Goog"])[1:]

  return (xData, yData)


# In[75]:

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


# In[76]:

import pandas as pd 
import numpy as np
import statsmodels.api as sm


# In[77]:

xData, yData = read_goog_sp500_logistic_data()


# In[78]:

logit = sm.Logit(yData, xData)


# In[79]:

# Fit the Logistic model
result = logit.fit()


# In[80]:

# All values >0.5 predict an up day for Google
predictions = (result.predict(xData) > 0.5)


# In[81]:

# Count the number of times the actual up days match the predicted up days
num_accurate_predictions = (list(yData == predictions)).count(True)


# In[82]:

pctAccuracy = float(num_accurate_predictions) / float(len(predictions))


# In[83]:

print "Accuracy: ", pctAccuracy


# In[84]:

import tensorflow as tf


# In[85]:

W = tf.Variable(tf.ones([1, 2]), name="W")
b = tf.Variable(tf.zeros([2]), name="b")


# In[86]:

x = tf.placeholder(tf.float32, [None, 1], name="x")


# In[87]:

y_ = tf.placeholder(tf.float32, [None, 2], name="y_")

y = tf.matmul(x, W) + b


# In[88]:

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# In[89]:

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[90]:

# All returns in a 2D array
# [[-0.02184618]
#  [ 0.00997998]
#  [ 0.04329069]
#  [ 0.03254923]
#  [-0.01781632]]
all_xs = np.expand_dims(xData[:,0], axis=1)


# In[91]:

# Another 2D array with 0 1 or 1 0 in each row
# 1 0 indicates a UP day
# 0 1 indicates a DOWN day
# [[0 1]
#  [1 0]
#  [1 0]
#  [1 0]
#  [1 0]]
all_ys = np.array([([1,0] if yEl == True else [0,1]) for yEl in yData])


# In[92]:

dataset_size = len(all_xs)


# In[93]:

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
        batch_start_idx = (i * batch_size) % (dataset_size)

      batch_end_idx = batch_start_idx + batch_size

      batch_xs = all_xs[batch_start_idx : batch_end_idx]
      batch_ys = all_ys[batch_start_idx : batch_end_idx]

      feed = { x: batch_xs, y_: batch_ys }

      sess.run(train_step, feed_dict=feed)

      if (i + 1) % 1000 == 0:
        print("After %d iteration:" % i)
        print(sess.run(W))
        print(sess.run(b))

        print("cross entropy: %f" % sess.run(cross_entropy, feed_dict=feed))

    # Test model
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy: %f" % sess.run(accuracy, feed_dict={x: all_xs, y_: all_ys}))


# In[94]:

trainWithMultiplePointsPerEpoch(20000, train_step, dataset_size)


# In[ ]:



