import numpy as np

# inputs = [[1, 2, 3, 2.5],
#           [2, 2.5,-1.7, 3],
#           [3.5, -1.9, -2.9, 3.2]]

# weights1 = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]

# biases1 = [2, 3, 0.6]

# weights2 = [[0.9, 2.8, -1.5],
#            [0.6, 1.91, -0.59],
#            [2.2, 0.97, -0.87]]
# biases2 = [2.5, 3.9, -0.6]


# layer1_output = np.dot(inputs, np.array(weights1).T) + biases1
# layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

# print(layer2_output)

X = [[1, 2, 3, 2.5],
     [2, 2.5,-1.7, 3],
     [3.5, -1.9, -2.9, 3.2]]

class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)  # it takes the number of inputs which is equal to the number of neurons in previous layer.

layer1.forward(X)
layer2.forward(layer1.output)

print(layer2.output)
