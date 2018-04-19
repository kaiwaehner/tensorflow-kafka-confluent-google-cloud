
# coding: utf-8

# In[ ]:

import tensorflow as tf
import google.datalab.ml as ml


# In[ ]:

original_image_list = ["./images/Oba.jpg", 
                       "./images/Moje.jpg"]


# In[ ]:

# Make a queue of file names including all the images specified.
filename_queue = tf.train.string_input_producer(original_image_list)


# In[ ]:

# Read an entire image file.
image_reader = tf.WholeFileReader()


# In[ ]:

with tf.Session() as sess:
    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = [];
    for i in range(len(original_image_list)):
        # Read a whole file from the queue, the first returned value in the tuple is the
        # filename which we are ignoring.
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as a JPEG file, this will turn it into a Tensor which we can
        # then use in training.
        image = tf.image.decode_jpeg(image_file)

        # Get a tensor of resized images.
        image = tf.image.resize_images(image, [224, 224])
        image.set_shape((224, 224, 3))

        # Get an image tensor and print its value.
        image_array = sess.run(image)
        print image_array.shape

        # The expand_dims adds a new dimension
        image_list.append(tf.expand_dims(image_array, 0))

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    index = 0

    # Write image summary
    summary_writer = tf.summary.FileWriter('./ImageReadAndResizeWithCoordinator', graph=sess.graph)

    for image_tensor in image_list:
        summary_str = sess.run(tf.summary.image("image-" + str(index), image_tensor))
        summary_writer.add_summary(summary_str)
        index += 1

    summary_writer.close()


# In[ ]:

tensorboard_pid = ml.TensorBoard.start('./ImageReadAndResizeWithCoordinator')


# In[ ]:

ml.TensorBoard.stop(tensorboard_pid)


# In[ ]:



