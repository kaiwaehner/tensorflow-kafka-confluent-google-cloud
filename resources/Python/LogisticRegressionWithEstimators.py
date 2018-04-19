
# coding: utf-8

# In[5]:

import pandas as pd 
import numpy as np


# In[6]:

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


# In[7]:

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


# In[8]:

def read_goog_sp500_data():
  """Returns a tuple with 2 fields, the returns for Google and the S&P 500.
  Each of the returns are in the form of a 1D array"""

  returns = read_goog_sp500_dataframe()

  # Filter out the very first row which does not have any value for returns
  xData = np.array(returns["SP500"])[1:]
  yData = np.array(returns["Goog"])[1:]

  return (xData, yData)


# In[43]:

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


# In[51]:

import pandas as pd 
import numpy as np
import statsmodels.api as sm


# In[52]:

xData, yData = read_goog_sp500_logistic_data()


# In[53]:

logit = sm.Logit(yData, xData)


# In[54]:

# Fit the Logistic model
result = logit.fit()


# In[55]:

# All values >0.5 predict an up day for Google
predictions = (result.predict(xData) > 0.5)


# In[56]:

# Count the number of times the actual up days match the predicted up days
num_accurate_predictions = (list(yData == predictions)).count(True)


# In[57]:

pctAccuracy = float(num_accurate_predictions) / float(len(predictions))


# In[58]:

print "Accuracy: ", pctAccuracy


# In[59]:

# Logistic regression with estimators

import tensorflow as tf


# In[60]:

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]


# In[62]:

estimator = tf.contrib.learn.LinearClassifier(feature_columns=features)


# In[63]:

# All returns in a 2D array
# [[-0.02184618]
# [ 0.00997998]
# [ 0.04329069]
# [ 0.03254923]
# [-0.01781632]]
x = np.expand_dims(xData[:,0], axis=1)


# In[64]:

# True/False values for up/down days in a 2D array
# [[False]
# [ True]
# [ True]
# [ True]
# [ True]
# [False]]
y = np.expand_dims(np.array(yData), axis=1)


# In[65]:

# Batch size of 100 and 10000 epochs
input_fn = tf.contrib.learn.io.numpy_input_fn({"x" : x}, y, batch_size=100, num_epochs=10000)


# In[69]:

fit = estimator.fit(input_fn=input_fn, steps=10000)


# In[68]:

# All data points in a single batch with just one epoch
input_fn_oneshot = tf.contrib.learn.io.numpy_input_fn({"x": x }, y, batch_size=len(x), num_epochs=1)


# In[70]:

results = fit.evaluate(input_fn=input_fn_oneshot, steps=1)


# In[71]:

print results


# In[72]:

for variable_name in fit.get_variable_names():
    print variable_name , " ---> " , fit.get_variable_value(variable_name)


# In[ ]:



