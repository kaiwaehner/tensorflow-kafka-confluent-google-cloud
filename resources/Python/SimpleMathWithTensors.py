
# coding: utf-8

# In[ ]:

import tensorflow as tf
import google.datalab.ml as ml


# In[ ]:

x = tf.constant([1000, 2000, 3000], name='x')
y = tf.constant([11, 222, 3333], name='y')


# In[ ]:

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")


# In[ ]:

final_div = tf.div(sum_x, prod_y, name="final_div")


# In[ ]:

final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")


# In[ ]:

sess = tf.Session()


# In[ ]:

print "x: ", sess.run(x)
print "y: ", sess.run(y)

print "sum(x): ", sess.run(sum_x)
print "prod(y): ", sess.run(prod_y)
print "sum(x) / prod(y):", sess.run(final_div)
print "mean(sum(x), prod(y)):", sess.run(final_mean)


# In[ ]:

writer = tf.summary.FileWriter('./SimpleMathWithTensors', sess.graph)
writer.close()

tensorboard_pid = ml.TensorBoard.start('./SimpleMathWithTensors')


# In[ ]:

ml.TensorBoard.stop(tensorboard_pid)


# In[ ]:



