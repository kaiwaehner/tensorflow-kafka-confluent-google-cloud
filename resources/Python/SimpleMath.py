
# coding: utf-8

# In[ ]:

import tensorflow as tf
import google.datalab.ml as ml


# In[ ]:

a = tf.constant(6.5, name='constant_a')

b = tf.constant(3.4, name='constant_b')
c = tf.constant(3.0, name='constant_c')

d = tf.constant(100.2, name='constant_d')


# In[ ]:

add = tf.add(a, b, name="add_ab")
subtract = tf.subtract(b, c, name="subtract_bc")
square = tf.square(d, name="square_d")


# In[ ]:

final_sum = tf.add_n([add, subtract, square], name="final_sum")


# In[ ]:

with tf.Session() as sess:

    print "a + b: ", sess.run(add)
    print "b - c: ", sess.run(subtract)
    print "Square of d: ", sess.run(square)
    
    print "Final sum", sess.run(final_sum)
    
    another_sum = tf.add_n([a, b, c, d, square], name="another_sum")
    print "Another sum ", sess.run(another_sum)
    


# In[ ]:

writer = tf.summary.FileWriter('./SimpleMath', sess.graph)
writer.close()


# In[ ]:

tensorboard_pid = ml.TensorBoard.start('./SimpleMath')


# In[ ]:

ml.TensorBoard.stop(tensorboard_pid)


# In[ ]:



