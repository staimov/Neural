import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, init_input_nodes, init_hidden_nodes, init_output_nodes, init_learning_rate):
        self.input_nodes = init_input_nodes
        self.hidden_nodes = init_hidden_nodes
        self.output_nodes = init_output_nodes

        self.learning_rate = init_learning_rate

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

        self.activation_function = lambda x: scipy.special.expit(x)  # сигмоида

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


def main():
    network = NeuralNetwork(3, 3, 3, 0.3)
    print(network.query([1.0, 0.5, -1.5]))


if __name__ == '__main__':
    # call the main function
    main()
