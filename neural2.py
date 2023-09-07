import numpy
import matplotlib.pyplot
import scipy.special
import imageio
import datetime
import scipy.ndimage

# train_file_name = "mnist\\mnist_train_100.csv"
# unzip .\mnist\mnist_train.zip first!
TRAIN_FILE_NAME = "mnist\\mnist_train.csv"
# test_file_name = "mnist\\mnist_test_10.csv"
TEST_FILE_NAME = "mnist\\mnist_test.csv"

INPUT_NODES = 784  # 28 * 28
HIDDEN_NODES = 200
OUTPUT_NODES = 10
LEARNING_RATE = 0.01
# EPOCHS is the number of times the training data set is used for training
EPOCHS = 10


class NeuralNetwork:
    def __init__(self, init_input_nodes, init_hidden_nodes, init_output_nodes, init_learning_rate):
        self.input_nodes = init_input_nodes
        self.hidden_nodes = init_hidden_nodes
        self.output_nodes = init_output_nodes

        self.learning_rate = init_learning_rate

        self.w_input2hidden = numpy.zeros((self.hidden_nodes, self.input_nodes))
        self.w_hidden2output = numpy.zeros((self.output_nodes, self.hidden_nodes))

        self.activation_function = lambda x: scipy.special.expit(x)  # сигмоида
        self.inverse_activation_function = lambda x: scipy.special.logit(x)  # обратная сигмоида

    def save_w_to_file(self):
        numpy.savetxt("w_input2hidden.csv", self.w_input2hidden, delimiter=',')
        numpy.savetxt("w_hidden2output.csv", self.w_hidden2output, delimiter=',')

    def init_w_from_file(self):
        self.w_input2hidden = numpy.loadtxt("w_input2hidden.csv", delimiter=',')
        self.w_hidden2output = numpy.loadtxt("w_hidden2output.csv", delimiter=',')

    def init_w_by_random(self):
        # начальные значения весовых коэффициентов получаем
        # как случайные величины с помощью нормального распределения
        self.w_input2hidden = numpy.random.normal(
            0.0,  # mu
            pow(self.input_nodes, -0.5),  # sigma
            (self.hidden_nodes, self.input_nodes))
        self.w_hidden2output = numpy.random.normal(
            0.0,  # mu
            pow(self.hidden_nodes, -0.5),  # sigma
            (self.output_nodes, self.hidden_nodes))

    def train(self, input_list, target_list):
        targets = numpy.array(target_list, ndmin=2).T
        outputs, hidden_outputs, inputs = self.query_helper(input_list)

        # output layer error is the (target - actual)
        output_errors = targets - outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.w_hidden2output.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.w_hidden2output += self.learning_rate * numpy.dot((output_errors * outputs * (1.0 - outputs)),
                                                               numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.w_input2hidden += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                              numpy.transpose(inputs))

    def query(self, input_list):
            final_outputs, hidden_outputs, inputs = self.query_helper(input_list)
            return final_outputs

    def query_helper(self, input_list):
        # convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.w_input2hidden, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.w_hidden2output, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs, hidden_outputs, inputs

    def back_query(self, target_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(target_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.w_hidden2output.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.w_input2hidden.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs


def make_target_list(output_nodes, marker):
    target_list = numpy.zeros(output_nodes) + 0.01
    target_list[marker] = 0.99
    return target_list


def scale_inputs(source, source_max):
    return (0.99 * source / source_max) + 0.01


def load_mist_file(file_name):
    input_lists = []
    markers = []
    with open(file_name, "r") as data_file:
        for line in data_file:
            values = line.split(",")
            markers.append(int(values[0]))
            input_lists.append(scale_inputs(numpy.asfarray(values[1:]), 255.0))

    data_file.close()
    return markers, input_lists


def input_list_from_image(image_file_name):
    image_matrix = imageio.imread(image_file_name, as_gray="F")
    image_list = 255.0 - image_matrix.reshape(784)
    return scale_inputs(image_list, 255.0)


def main():
    network = NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES, LEARNING_RATE)
    # network.init_w_by_random()
    load(network)

    # test(network, False)

    # train(network)
    # save(network)

    # test(network, False)
    #
    back_test(network, 0)
    back_test(network, 1)
    back_test(network, 2)
    back_test(network, 3)
    back_test(network, 4)
    back_test(network, 5)
    back_test(network, 6)
    back_test(network, 7)
    back_test(network, 8)
    back_test(network, 9)

    # test_single(network, 6)
    #
    # test_user_image(network, "my_images\\0-99.png", 0)
    # test_user_image(network, "my_images\\1-99.png", 1)
    # test_user_image(network, "my_images\\2-99.png", 2)
    # test_user_image(network, "my_images\\3-99.png", 3)
    # test_user_image(network, "my_images\\4-99.png", 4)
    # test_user_image(network, "my_images\\5-99.png", 5)
    # test_user_image(network, "my_images\\6-02.png", 6)
    # test_user_image(network, "my_images\\7-99.png", 7)
    # test_user_image(network, "my_images\\8-99.png", 8)
    # test_user_image(network, "my_images\\9-99.png", 9)
    #
    # test_user_image(network, "my_images\\3-01.png", 3)
    # test_user_image(network, "my_images\\3-02.png", 3)
    # test_user_image(network, "my_images\\4-01.png", 4)
    # test_user_image(network, "my_images\\4-02.png", 4)
    # test_user_image(network, "my_images\\4-03.png", 4)
    # test_user_image(network, "my_images\\4-04.png", 4)
    # test_user_image(network, "my_images\\5-01.png", 5)
    # test_user_image(network, "my_images\\6-01.png", 6)
    # test_user_image(network, "my_images\\6-99.png", 6)
    # test_user_image(network, "my_images\\6-03.png", 6)
    # test_user_image(network, "my_images\\7-01.png", 7)
    # test_user_image(network, "my_images\\7-02.png", 7)
    # test_user_image(network, "my_images\\7-03.png", 7)
    # test_user_image(network, "my_images\\7-04.png", 7)


def train(network):
    start = datetime.datetime.now()
    print(f"training ({start})...")
    mnist_train_helper(network, EPOCHS, True)
    finish = datetime.datetime.now()
    delta = finish - start
    print(delta)


def test(network, display):
    print("testing...")
    performance = mnist_test_helper(network, display)
    print(f"performance = {performance:.4f}")


def back_test(network, marker):
    print("back testing...")
    targets = numpy.zeros(OUTPUT_NODES) + 0.01
    targets[marker] = 0.99
    print(f"marker={marker}")
    print(targets)
    image_data = network.back_query(targets)
    matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()


def test_single(network, index):
    print("testing...")
    mnist_single_test_helper(network, index)


def test_user_image(network, image_file_name, marker):
    print("testing...")
    input_list = input_list_from_image(image_file_name)
    input_lists = [input_list]
    markers = [marker]
    mnist_single_test_helper2(network, 0, markers, input_lists)


def save(network):
    print(f"saving...")
    network.save_w_to_file()


def load(network):
    print("loading...")
    network.init_w_from_file()


def mnist_train_helper(network, epochs, with_rotation):
    markers, input_lists = load_mist_file(TRAIN_FILE_NAME)

    for e in range(epochs):
        index = 0
        while index < len(input_lists):
            target_list = make_target_list(network.output_nodes, markers[index])
            network.train(input_lists[index], target_list)

            if with_rotation:
                # create rotated variations

                # rotated anticlockwise by 10 degrees
                inputs_plusx_img = scipy.ndimage.interpolation.rotate(input_lists[index].reshape(28, 28), 10,
                                                                      cval=0.01, order=1, reshape=False)
                network.train(inputs_plusx_img.reshape(784), target_list)

                # rotated clockwise by 10 degrees
                inputs_minusx_img = scipy.ndimage.interpolation.rotate(input_lists[index].reshape(28, 28), -10,
                                                                       cval=0.01, order=1, reshape=False)
                network.train(inputs_minusx_img.reshape(784), target_list)

            index += 1


def mnist_test_helper(network, display):
    markers, input_lists = load_mist_file(TEST_FILE_NAME)
    scorecard = []

    index = 0
    while index < len(input_lists):
        outputs = network.query(input_lists[index])
        result = numpy.argmax(outputs)

        if display:
            print(f"marker={markers[index]} result={result} success={result == markers[index]}")
            matplotlib.pyplot.imshow(input_lists[index].reshape((28, 28)), cmap="Greys", interpolation="None")
            matplotlib.pyplot.show()

        if result == markers[index]:
            scorecard.append(1)
        else:
            scorecard.append(0)

        index += 1

    scorecard_array = numpy.asarray(scorecard)
    success_sum = scorecard_array.sum()
    return success_sum / scorecard_array.size


def mnist_single_test_helper(network, index):
    markers, input_lists = load_mist_file(TEST_FILE_NAME)
    return mnist_single_test_helper2(network, index, markers, input_lists)


def mnist_single_test_helper2(network, index, markers, input_lists):
    outputs = network.query(input_lists[index])
    result = numpy.argmax(outputs)
    success = (result == markers[index])

    count = 0
    while count < len(outputs):
        print(count, outputs[count])
        count += 1
    print(f"marker={markers[index]} result={result} success={success}")
    matplotlib.pyplot.imshow(input_lists[index].reshape((28, 28)), cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()

    return success


if __name__ == '__main__':
    # call the main function
    main()
