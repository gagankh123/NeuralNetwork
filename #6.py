# all the output conversion is because we need to remove all the -ve values as in #5.py all negatives are 0.
# To remove all -ves we can not do absolute otherwise there will not be any difference for +ve and -ve values.

'''
import math

layer_outputs = [4.8, 1.21, 2.385]
# E = 2.71828182846
E = math.e
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output) # exponential value

print(exp_values)

# normalisation
norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value/norm_base)

print(norm_values)
print(sum(norm_values))
'''

import math
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# layer_outputs = [[4.8, 1.21, 2.385],
#                  [1.8, 4.26, 2.385],
#                  [3.9, 2.22, 0.385]]

# exp_values = np.exp(layer_outputs)
# print(np.sum(exp_values, axis=1, keepdims=True))
# # E = math.e
# # exp_values = np.exp(layer_outputs)
# # norm_base = sum(exp_values)
# norm_values = exp_values/np.sum(exp_values)
# print(norm_values)
# print(sum(norm_values))

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probability = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probability

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = ActivationReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)