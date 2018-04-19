
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[2]:

import os
import collections
import itertools


# In[3]:

import numpy as np
import pandas as pd
import tensorflow as tf


# In[4]:

from six.moves import urllib


# In[5]:

print(np.__version__)
print(pd.__version__)
print(tf.__version__)


# In[6]:

URL_PATH = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

DOWNLOADED_FILENAME = "automobiles.csv"

def download_data():
    if not os.path.exists(DOWNLOADED_FILENAME):
        filename, _ = urllib.request.urlretrieve(URL_PATH, DOWNLOADED_FILENAME)

    print('Found and verified file from this path: ', URL_PATH)
    print('Downloaded file: ', DOWNLOADED_FILENAME)        


# In[7]:

download_data()


# In[8]:

COLUMN_TYPES = collections.OrderedDict([
    ("symboling", int),
    ("normalized-losses", float),
    ("make", str),
    ("fuel-type", str),
    ("aspiration", str),
    ("num-of-doors", str),
    ("body-style", str),
    ("drive-wheels", str),
    ("engine-location", str),
    ("wheel-base", float),
    ("length", float),
    ("width", float),
    ("height", float),
    ("curb-weight", float),
    ("engine-type", str),
    ("num-of-cylinders", str),
    ("engine-size", float),
    ("fuel-system", str),
    ("bore", float),
    ("stroke", float),
    ("compression-ratio", float),
    ("horsepower", float),
    ("peak-rpm", float),
    ("city-mpg", float),
    ("highway-mpg", float),
    ("price", float)
])


# In[9]:

df = pd.read_csv(DOWNLOADED_FILENAME, names=COLUMN_TYPES.keys(),
                 dtype=COLUMN_TYPES, na_values="?")


# In[10]:

df.head()


# In[11]:

df.count()


# In[12]:

df = df.dropna()


# In[13]:

df.count()


# In[14]:

TRIMMED_CSV_COLUMNS = [
    "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
    "drive-wheels", "curb-weight", "engine-type", "num-of-cylinders", "engine-size",
    "fuel-system", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]


# In[15]:

df = df[TRIMMED_CSV_COLUMNS]


# In[16]:

df.head()


# In[17]:

Y_NAME = "price"

def get_training_test_prediction_data(df):
    
    # Generate a unique shuffle each time
    np.random.seed(None)

    # Split the data into train/test subsets.
    x_train = df.sample(frac=0.8, random_state=None)
    
    # Remove the training data from the original dataset
    x_test = df.drop(x_train.index)
    
    # Choose a small sample from the test data for prediction
    x_predict = x_test.sample(frac=0.2, random_state=None)
    
    # Extract the label from the features DataFrame.
    y_train = x_train.pop(Y_NAME)
    y_test = x_test.pop(Y_NAME)
    y_predict = x_predict.pop(Y_NAME)
    
    return (x_train, y_train), (x_test, y_test), (x_predict, y_predict) 


# In[19]:

(x_train, y_train), (x_test, y_test), (x_predict, y_predict) =     get_training_test_prediction_data(df)


# In[20]:

x_train.head()


# In[21]:

y_train.head()


# In[22]:

PRICE_SCALING_FACTOR = 10000

y_train /= PRICE_SCALING_FACTOR
y_test /= PRICE_SCALING_FACTOR


# In[23]:

y_train.head()


# In[24]:

df['make'].unique()


# In[25]:

df['fuel-type'].unique()


# In[26]:

df['aspiration'].unique()


# In[27]:

df['num-of-doors'].unique()


# In[28]:

df['body-style'].unique()


# In[29]:

df['drive-wheels'].unique()


# In[30]:

df['engine-type'].unique()


# In[31]:

df['num-of-cylinders'].unique()


# In[32]:

df['fuel-system'].unique()


# In[33]:

curb_weight = tf.feature_column.numeric_column("curb-weight")

engine_size = tf.feature_column.numeric_column("engine-size")

horsepower = tf.feature_column.numeric_column("horsepower")

peak_rpm = tf.feature_column.numeric_column("peak-rpm")

city_mpg = tf.feature_column.numeric_column("city-mpg")

highway_mpg = tf.feature_column.numeric_column("highway-mpg")


# In[34]:

body_style = tf.feature_column.categorical_column_with_vocabulary_list(
      key="body-style", vocabulary_list=df['body-style'].unique())

fuel_type = tf.feature_column.categorical_column_with_vocabulary_list(
      key="fuel-type", vocabulary_list=df['fuel-type'].unique())

aspiration = tf.feature_column.categorical_column_with_vocabulary_list(
      key="aspiration", vocabulary_list=df['aspiration'].unique())

num_of_doors = tf.feature_column.categorical_column_with_vocabulary_list(
      key="num-of-doors", vocabulary_list=df['num-of-doors'].unique())

drive_wheels = tf.feature_column.categorical_column_with_vocabulary_list(
      key="drive-wheels", vocabulary_list=df['drive-wheels'].unique())

engine_type = tf.feature_column.categorical_column_with_vocabulary_list(
      key="engine-type", vocabulary_list=df['engine-type'].unique())

num_of_cylinders = tf.feature_column.categorical_column_with_vocabulary_list(
      key="num-of-cylinders", vocabulary_list=df['num-of-cylinders'].unique())

fuel_system = tf.feature_column.categorical_column_with_vocabulary_list(
      key="fuel-system", vocabulary_list=df['fuel-system'].unique())


# In[35]:

make = tf.feature_column.categorical_column_with_hash_bucket(
      key="make", hash_bucket_size=50)


# In[36]:

feature_columns = [
    curb_weight, engine_size, horsepower, peak_rpm, city_mpg, highway_mpg,

    tf.feature_column.indicator_column(body_style),

    tf.feature_column.embedding_column(fuel_type, dimension=3),

    tf.feature_column.embedding_column(aspiration, dimension=3),
    tf.feature_column.embedding_column(num_of_doors, dimension=3),
    tf.feature_column.embedding_column(drive_wheels, dimension=3),
    tf.feature_column.embedding_column(engine_type, dimension=3),
    tf.feature_column.embedding_column(num_of_cylinders, dimension=3),
    tf.feature_column.embedding_column(fuel_system, dimension=3),

    tf.feature_column.embedding_column(make, dimension=4)    
]


# In[37]:

def input_fn(x_data, y_data, num_epochs, shuffle):

    return tf.estimator.inputs.pandas_input_fn(
          x=x_data,
          y=y_data,
          batch_size=64,
          num_epochs=num_epochs,
          shuffle=shuffle)            


# In[69]:

model = tf.estimator.DNNRegressor(
      hidden_units=[24, 16, 24], feature_columns=feature_columns)


# In[70]:

model.train(input_fn=input_fn(x_train, y_train, num_epochs=None, shuffle=True), steps=20000)


# In[71]:

results = model.evaluate(input_fn=input_fn(x_test, y_test, num_epochs=1, shuffle=False))


# In[72]:

for key in sorted(results):
    print("%s: %s" % (key, results[key]))


# In[73]:

average_loss = results["average_loss"]


# In[74]:

print("\nRMS error for the test set: ${:.0f}"
        .format(PRICE_SCALING_FACTOR * average_loss**0.5))


# In[75]:

len(x_predict), len(y_predict)


# In[76]:

predict_results = model.predict(input_fn=input_fn(x_predict, y_predict, num_epochs=1, shuffle=False))


# In[77]:

predictions = list(itertools.islice(predict_results, len(x_predict)))


# In[78]:

predictions


# In[79]:

predicted_prices = [obj['predictions'][0] * PRICE_SCALING_FACTOR for obj in predictions]


# In[80]:

predicted_prices


# In[81]:

compare_df = x_predict.copy()


# In[82]:

compare_df


# In[83]:

compare_df['actual-price'] = y_predict
compare_df['predicted-price'] = predicted_prices


# In[84]:

compare_df


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



