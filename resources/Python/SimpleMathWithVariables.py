
# coding: utf-8

# In[ ]:

import tensorflow as tf
import google.datalab.ml as ml


# In[ ]:


# y = Wx + b
W = tf.Variable([2.5, 4.0], tf.float32, name='var_W')

x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([5.0, 10.0], tf.float32, name='var_b')

y = W * x + b


# In[ ]:

# Initialize all variables defined
init = tf.global_variables_initializer()


# In[ ]:

with tf.Session() as sess:
    sess.run(init)

    print "Final result: Wx + b = ", sess.run(y, feed_dict={x: [10, 100]})


# In[ ]:

init = tf.variables_initializer([W])


# In[ ]:

with tf.Session() as sess:
    sess.run(init)

    print "Final result: Wx + b = ", sess.run(y, feed_dict={x: [10, 100]})


# In[ ]:

number = tf.Variable(10)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()


# In[ ]:

result = number.assign(tf.multiply(number, multiplier))


# In[ ]:

with tf.Session() as sess:
    sess.run(init)

    for i in range(5):
        print "Result number * multiplier = ", sess.run(result)
        print "Increment multiplier, new value = ", sess.run(multiplier.assign_add(1))


# In[ ]:

writer = tf.summary.FileWriter('./SimpleMathWithVariables', sess.graph)
writer.close()

tensorboard_pid = ml.TensorBoard.start('./SimpleMathWithVariables')


# In[ ]:

ml.TensorBoard.stop(tensorboard_pid)

