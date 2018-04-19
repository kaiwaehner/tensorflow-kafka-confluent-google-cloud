
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# In[3]:

import collections
import math
import os
import random
import re


# In[4]:

from six.moves import urllib


# In[5]:

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf


# In[6]:

print(np.__version__)
print(mp.__version__)
print(tf.__version__)


# In[7]:

# http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz

def get_reviews(path, positive=True):
    
    label = 1 if positive else 0

    reviews = []
    labels = []

    with open(path, 'r') as f:
        reviews = f.readlines()

    for review in reviews:
        labels.append(label)

    return reviews, labels        


# In[8]:

def extract_labels_data():
    
    # This code assumes that the files rt-polarity.pos and rt-polarity.neg have already
    # been downloaded and are in the current working directory

    positive_reviews, positive_labels = get_reviews("rt-polarity.pos", positive=True)

    negative_reviews, negative_labels = get_reviews("rt-polarity.neg", positive=False)

    data = positive_reviews + negative_reviews
    labels = positive_labels + negative_labels

    return labels, data    


# In[10]:

labels, data = extract_labels_data()


# In[11]:

labels[:5]


# In[12]:

data[:5]


# In[13]:

len(labels), len(data)


# In[14]:

max_document_length = max([len(x.split(" ")) for x in data])


# In[15]:

print(max_document_length)


# In[16]:

MAX_SEQUENCE_LENGTH = 50

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SEQUENCE_LENGTH)


# In[17]:

x_data = np.array(list(vocab_processor.fit_transform(data)))


# In[18]:

y_output = np.array(labels)


# In[19]:

vocabulary_size = len(vocab_processor.vocabulary_)
print(vocabulary_size)


# In[20]:

data[3:5]


# In[21]:

x_data[3:5]


# In[22]:

y_output[:5]


# In[23]:

np.random.seed(22)
shuffle_indices = np.random.permutation(np.arange(len(x_data)))

x_shuffled = x_data[shuffle_indices]
y_shuffled = y_output[shuffle_indices]


# In[24]:

TRAIN_DATA = 9000
TOTAL_DATA = len(labels)

train_data = x_shuffled[:TRAIN_DATA]
train_target = y_shuffled[:TRAIN_DATA]

test_data = x_shuffled[TRAIN_DATA:TOTAL_DATA]
test_target = y_shuffled[TRAIN_DATA:TOTAL_DATA]


# In[25]:

tf.reset_default_graph()


# In[26]:

x = tf.placeholder(tf.int32, [None, MAX_SEQUENCE_LENGTH])
y = tf.placeholder(tf.int32, [None])


# In[27]:

batch_size = 25
embedding_size = 50
max_label = 2


# In[28]:

embedding_matrix = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))


# In[29]:

embeddings = tf.nn.embedding_lookup(embedding_matrix, x)


# In[30]:

embeddings


# In[31]:

embedding_matrix


# In[32]:

lstmCell = tf.contrib.rnn.BasicLSTMCell(embedding_size)


# In[33]:

lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)


# In[34]:

_, (encoding, _) = tf.nn.dynamic_rnn(lstmCell, embeddings, dtype=tf.float32)


# In[35]:

encoding


# In[36]:

logits = tf.layers.dense(encoding, max_label, activation=None)


# In[37]:

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)


# In[38]:

loss = tf.reduce_mean(cross_entropy)


# In[39]:

prediction = tf.equal(tf.argmax(logits, 1), tf.cast(y, tf.int64))


# In[40]:

accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))


# In[41]:

optimizer = tf.train.AdamOptimizer(0.01)
train_step = optimizer.minimize(loss)


# In[42]:

init = tf.global_variables_initializer()


# In[43]:

num_epochs = 20


# In[44]:

with tf.Session() as session:
    init.run()

    for epoch in range(num_epochs):

        num_batches = int(len(train_data) // batch_size) + 1

        for i in range(num_batches):

            # Select train data
            min_ix = i * batch_size
            max_ix = np.min([len(train_data), ((i+1) * batch_size)])

            x_train_batch = train_data[min_ix:max_ix]
            y_train_batch = train_target[min_ix:max_ix]

            train_dict = {x: x_train_batch, y: y_train_batch}
            session.run(train_step, feed_dict=train_dict)

            train_loss, train_acc = session.run([loss, accuracy], feed_dict=train_dict)

        test_dict = {x: test_data, y: test_target}
        test_loss, test_acc = session.run([loss, accuracy], feed_dict=test_dict)    
            
        print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.5}'.format(epoch + 1, test_loss, test_acc)) 
            


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



