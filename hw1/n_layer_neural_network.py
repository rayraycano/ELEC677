# Author: Raymond
# ELEC677 Fall 2016
from three_layer_neural_network import NeuralNetwork, generate_data
import numpy as np
from Utils import *

class DeepNeuralNetwork(NeuralNetwork):

    def __init__(self, nn_input_dim, nn_output_dim, num_hlayers, layer_sizes, actFun_type='tanh', reg_lambda=.01, seed=0):
        """
        :param nn_input_dim: input dimension
        :param nn_output_dim: output dimension
        :param num_hlayers: number of hidden layers in the net
        :param layer_sizes: size of each hidden layer
        :param actFun_type: type of activation function being used
        :param reg_lambda: lambda value for regularizaiton
        :param seed: random seed
        """
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.num_hlayers = num_hlayers
        # If the number of layers provided is smaller than the layer sizes, ignore the layer sizes.
        try:
            self.layer_sizes = layer_sizes[:num_hlayers]
        except IndexError:
            raise "A size needs to be provided for each hidden layer. Not enough sizes were provided"
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        np.random.seed(seed)
        self.num_layers = num_hlayers + 2

        # Append the input and output dimensions to our layers
        self.layer_sizes.insert(0, nn_input_dim)
        self.layer_sizes.append(nn_output_dim)

        self.layers = self.init_layers()

    def init_layers(self):
        """
        Initialize the layers of the NeuralNetwork
        :return: an array of Layer object initialized with proper dimensions and
        activation type.
        """
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layer = Layer(self.layer_sizes[i], self.layer_sizes[i+1], self.actFun_type)
            layers.append(layer)
        return layers

    def feedforward(self, X, actFun):
        """
        Perform the forward pass across the neural network
        :param X: original input
        :param actFun: activation funciton
        :return: Nothing. Sets the probs parameter of the net.
        """

        data = X
        for i in range(len(self.layers) - 1):
            data = self.layers[i].feedforward(data)
            self.layers[i+1].prev_z = self.layers[i].z
        z_last = self.layers[-1].feedforward(data, middle_layer=False)
        self.get_probs(z_last)

    def backprop(self, X, y):
        num_examples = len(X)
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1

        dout = delta3
        if dout is None:
            print "YIKES"
        for i in range(len(self.layers) - 1, 0, -1):
            dout = self.layers[i].backprop(dout)
        _ = self.layers[0].backprop(dout, first_layer=True)

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE

        y_expanded = np.vstack((np.logical_not(y.astype(bool)).astype(int), y)).T

        data_loss = -1 * np.sum(y_expanded * np.log(self.probs))

        data_loss += self.reg_lambda / 2. * np.sum([np.sum(np.square(x.W)) for x in self.layers])

        return (1./num_examples) * data_loss

    def fit_model(self, X, y, epsilon=.01, num_passes=10000, print_loss=True):
        """
        Fit the model to the X, y data given
        :param X: input data
        :param y: output labels
        :param epsilon: learning rate
        :param num_passes: number of passes through the data
        :param print_loss: Boolean to instruct whether losses are printed
        :return: Nothing
        """
        for i in range(num_passes):
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            self.backprop(X, y)
            self.update_weights(epsilon)
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


    def update_weights(self, epsilon):
        for layer in self.layers:
            layer.update_weights_and_biases(epsilon, self.reg_lambda)


class Layer():
        def __init__(self, input_dim, output_dim, actFun_type):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.actFun_type = actFun_type
            self.z = None
            self.a = None
            self.X = None
            self.W = random_init(input_dim, output_dim)
            self.b = np.zeros((1, output_dim))
            self.dW = None
            self. db = None
            self.prev_z = None

        def feedforward(self, X, middle_layer=True):
            self.X = X
            self.z = NeuralNetwork.affine_forward(X, self.W, self.b)
            if middle_layer:
                self.a = NeuralNetwork.actFun(self.z, self.actFun_type)
            out = self.a if middle_layer else self.z
            return out

        def backprop(self, delta_out, first_layer=False):
            # if self.prev_z is None:
            #     print "prev_z none"
            self.dW, self.db, dX = NeuralNetwork.affine_backwards(delta_out, self.X, self.W, self.prev_z, self.actFun_type,
                                                                  first_layer=first_layer)
            # if dX is None:
            #     print "NONEHERE"
            # else:
            #     print 'good'
            return dX

        def update_weights_and_biases(self, epsilon, reg):
            self.dW += reg * self.W
            self.W += - epsilon * self.dW
            self.b += - epsilon * self.db



def main():
    # # generate and visualize Make-Moons dataset
    X, y = generate_data()
    
    model = DeepNeuralNetwork(nn_input_dim=2, nn_output_dim=2, num_hlayers=5, layer_sizes=[4,8,16,32,64], actFun_type='tanh')
    model.fit_model(X,y, epsilon=.01, num_passes=50000)
    model.visualize_decision_boundary(X, y)

if __name__ == "__main__":
    main()
