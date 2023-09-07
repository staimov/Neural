import numpy
import matplotlib.pyplot
import scipy.ndimage


def make_targets(output_nodes, marker):
    targets = numpy.zeros(output_nodes) + 0.01
    targets[marker] = 0.99
    return targets


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


def main():
    markers, input_lists = load_mist_file("mnist\\mnist_test_10.csv")

    index = 2
    print(markers[index])
    print(input_lists[index])
    print(make_targets(10, markers[index]))
    matplotlib.pyplot.imshow(input_lists[index].reshape((28, 28)), cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()

    inputs_plusx_img = scipy.ndimage.interpolation.rotate(input_lists[index].reshape(28, 28), 10,
                                                          cval=0.01, order=1, reshape=False)
    matplotlib.pyplot.imshow(inputs_plusx_img.reshape((28, 28)), cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()

    # rotated clockwise by 10 degrees
    inputs_minusx_img = scipy.ndimage.interpolation.rotate(input_lists[index].reshape(28, 28), -10,
                                                           cval=0.01, order=1, reshape=False)
    matplotlib.pyplot.imshow(inputs_minusx_img.reshape((28, 28)), cmap="Greys", interpolation="None")
    matplotlib.pyplot.show()


if __name__ == '__main__':
    # call the main function
    main()
