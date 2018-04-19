
# coding: utf-8

# In[27]:

import tensorflow as tf


# In[28]:

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)


# In[29]:

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b


# In[30]:

y = tf.placeholder(tf.float32)


# In[31]:

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


# In[32]:

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]


# In[33]:

# training loop
init = tf.global_variables_initializer()


# In[34]:

with tf.Session() as sess:
  sess.run(init)
  
  for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

  # evaluate training accuracy
  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


# In[ ]:



