# https://adventuresinmachinelearning.com/python-tensorflow-tutorial/

import imageio
import matplotlib.pyplot
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow_datasets


# from tensorflow.examples.tutorials.mnist import input_data


def main():
    tf.disable_v2_behavior()

    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    mnist = tensorflow_datasets.load('mnist')
    save_path = ".\\model.ckpt"
    train = True

    # Python optimisation variables
    hidden_nodes = 300
    learning_rate = 0.5
    epochs = 10
    batch_size = 100

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(tf.float32, [None, 10], name='y')

    # now declare the weights connecting the input to the hidden layer
    w1 = tf.Variable(tf.random_normal([784, hidden_nodes], stddev=0.03), name='w1')
    b1 = tf.Variable(tf.random_normal([hidden_nodes]), name='b1')
    # and the weights connecting the hidden layer to the output layer
    w2 = tf.Variable(tf.random_normal([hidden_nodes, 10], stddev=0.03), name='w2')
    b2 = tf.Variable(tf.random_normal([10]), name='b2')

    # calculate the output of the hidden layer
    hidden_out = tf.add(tf.matmul(x, w1), b1)
    hidden_out = tf.nn.relu(hidden_out)

    # now calculate the output of the output layer -
    # in this case, let's use a softmax activated output layer
    y_out = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))

    y_clipped = tf.clip_by_value(y_out, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    result = tf.argmax(y_out, 1, name='result')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_out, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    if train:
        # start the session
        with tf.Session() as session:
            # creating the writer inside the session
            # writer = tf.summary.FileWriter("./graphs", session.graph)
            # tensorboard --logdir=./TensorFlow/graphs --port=6006
            # http://localhost:6006/#graphs&run=.

            # initialise the variablespython
            session.run(init_op)
            total_batch = int(len(mnist.train.labels) / batch_size)  # error here if tf2 used
            for epoch in range(epochs):
                avg_cost = 0
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                    dummy, c = session.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

            print(session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
            # Save the variables to disk.
            saver.save(session, save_path)

    # test
    with tf.Session() as session:
        # Restore variables from disk.
        saver.restore(session, save_path)

        for marker in range(10):
            test_user_image(session, f"..\\my_images\\{marker}-99.png", marker)

        test_user_image(session, "..\\my_images\\3-01.png", 3)
        test_user_image(session, "..\\my_images\\3-02.png", 3)
        test_user_image(session, "..\\my_images\\4-01.png", 4)
        test_user_image(session, "..\\my_images\\4-02.png", 4)
        test_user_image(session, "..\\my_images\\4-03.png", 4)
        test_user_image(session, "..\\my_images\\4-04.png", 4)
        test_user_image(session, "..\\my_images\\5-01.png", 5)
        test_user_image(session, "..\\my_images\\6-01.png", 6)
        test_user_image(session, "..\\my_images\\6-02.png", 6)
        test_user_image(session, "..\\my_images\\6-03.png", 6)
        test_user_image(session, "..\\my_images\\7-01.png", 7)
        test_user_image(session, "..\\my_images\\7-02.png", 7)
        test_user_image(session, "..\\my_images\\7-03.png", 7)
        test_user_image(session, "..\\my_images\\7-04.png", 7)


def test_user_image(session, file_name, marker):
    result = session.graph.get_tensor_by_name("result:0")
    x = session.graph.get_tensor_by_name("x:0")
    y = session.graph.get_tensor_by_name("y:0")
    label, input_list = input_list_from_image(file_name, marker)
    # print(input_list)
    # print(label)
    print(f"marker={marker} result={session.run(result[0], feed_dict={x: [input_list], y: [label]})}")
    matplotlib.pyplot.imshow(input_list.reshape(28, 28), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()


def input_list_from_image(image_file_name, marker):
    image_matrix = imageio.imread(image_file_name, as_gray="F")
    image_list = 255.0 - image_matrix.reshape(784)
    input_list = scale_inputs(image_list, 255.0)
    label = np.zeros(10)
    label[marker] = 1
    return label, input_list


def scale_inputs(source, source_max):
    return 0.9999999 * source / source_max


if __name__ == '__main__':
    # call the main function
    main()
