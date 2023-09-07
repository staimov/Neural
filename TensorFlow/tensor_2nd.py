#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

# a = (b + c) ∗ (c + 2)

tf.disable_v2_behavior()

# first, create a TensorFlow constant
const = tf.constant(2.0, name="const")

# create TensorFlow variables
b = tf.placeholder(tf.float32, [None, 1], name='b')
c = tf.Variable(1.0, name='c')

# now create some operations
d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

# setup the variable initialisation
init_op = tf.global_variables_initializer()

# start the session
with tf.Session() as session:
    # creating the writer inside the session
    writer = tf.summary.FileWriter("./graphs", session.graph)
    # tensorboard --logdir=./TensorFlow/graphs --port=6006
    # http://localhost:6006/#graphs&run=.

    # initialise the variables
    session.run(init_op)
    # compute the output of the graph
    a_out = session.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    print("Variable a is {}".format(a_out))
