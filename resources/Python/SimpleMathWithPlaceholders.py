
# coding: utf-8

# In[ ]:

import tensorflow as tf
import google.datalab.ml as ml


# In[ ]:

x = tf.placeholder(tf.int32, shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3], name='y')


# In[ ]:

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")


# In[ ]:

final_div = tf.div(sum_x, prod_y, name="final_div")


# In[ ]:

with tf.Session() as sess:

    print "sum(x): ", sess.run(sum_x, feed_dict={x: [100, 200, 300]})
    print "prod(y): ", sess.run(prod_y, feed_dict={y: [11, 22, 33]})

    # This needs both x and y placeholder values for its calculation
    print "sum(x) / prod(y):", sess.run(final_div, feed_dict={x: [100, 200, 300], y: [1, 2, 3]})


# In[ ]:

writer = tf.summary.FileWriter('./SimpleMathWithPlaceholders', sess.graph)
writer.close()

tensorboard_pid = ml.TensorBoard.start('./SimpleMathWithPlaceholders')


# In[ ]:

ml.TensorBoard.stop(tensorboard_pid)

