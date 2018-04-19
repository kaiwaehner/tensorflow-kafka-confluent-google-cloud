
# coding: utf-8

# In[ ]:

import pandas as pd 
import numpy as np


# In[ ]:

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


# In[ ]:

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


# In[ ]:

def read_goog_sp500_data():
  """Returns a tuple with 2 fields, the returns for Google and the S&P 500.
  Each of the returns are in the form of a 1D array"""

  returns = read_goog_sp500_dataframe()

  # Filter out the very first row which does not have any value for returns
  xData = np.array(returns["SP500"])[1:]
  yData = np.array(returns["Goog"])[1:]

  return (xData, yData)


# In[ ]:

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


# In[ ]:

from sklearn import datasets, linear_model


# In[ ]:

xData, yData = read_goog_sp500_data()


# In[ ]:

googModel = linear_model.LinearRegression()
googModel.fit(xData.reshape(-1,1), yData.reshape(-1,1))


# In[ ]:

print googModel.coef_
print googModel.intercept_


# In[ ]:

import tensorflow as tf


# In[ ]:

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]


# In[ ]:

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)


# In[ ]:

input_fn = tf.contrib.learn.io.numpy_input_fn({"x" : xData}, yData, batch_size=len(xData),
                                              num_epochs=10000)


# In[ ]:

fit = estimator.fit(input_fn=input_fn, steps=10000)


# In[ ]:

for variable_name in fit.get_variable_names():
   print variable_name , " ---> " , fit.get_variable_value(variable_name)


# In[ ]:



