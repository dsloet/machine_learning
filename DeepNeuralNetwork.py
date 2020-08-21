import numpy as np

class DNNClassifier():
    def __init__(self, epochs = 1000, learning_rate = 0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate

    def initialise_nn(self, nn_architecture, seed=99):
        self.nn_architecture = nn_architecture
        self.seed = seed

        np.random.seed(self.seed)
        number_of_layers = len(self.nn_architecture)
        self.params_values = {}

        for layer_idx, layer in enumerate(nn_architecture, 1):
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            self.params_values['W' + str(layer_idx)] = np.random.randn(
                layer_input_size, layer_input_size) * 0.1

