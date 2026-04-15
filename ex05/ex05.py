import sys
sys.path.append('../ex02')
from ex02 import OurNeuralNetwork
import numpy as np

class Neuron:
    def __init__(self, weight1, weight2, bias, regression=False):
        self.weights_1 = weight1
        self.weights_2 = weight2
        self.bias = bias
        self.regression = regression

    def feedforward(self, x1, x2):
        total = (x1 * self.weights_1) + (x2 * self.weights_2) + self.bias
        if self.regression:
            return total                       # identity: output as-is
        return 1 / (1 + np.exp(-total))        # sigmoid


# Q1
neuron = Neuron(0, 1, 4, regression=True)
print(neuron.feedforward(2, 3))  # Expected: 7



# Q2
students = {'Bob': (12, 15), 'Eli': (10, 9), 'Tom': (18, 18), 'Ryan': (13, 14)}
y_true = np.array([16, 10, 19, 16])


neuron_h1 = Neuron(0.05, 0.001, 0, regression=False)
neuron_h2 = Neuron(0.002, 0.003, 0, regression=False)
neuron_o1 = Neuron(2, 7, 10, regression=True)


network = OurNeuralNetwork(neuron_h1, neuron_h2, neuron_o1)


y_pred = []
for name, (math, chem) in students.items():
    output = network.feedforward(math, chem)
    y_pred.append(output)
    print(f'{name}: {output}')

# Q3: MSE
y_pred = np.array(y_pred)
mse = np.mean((y_true - y_pred) ** 2)
print(f'MSE: {mse}')  # Expected: 10.23760869990

