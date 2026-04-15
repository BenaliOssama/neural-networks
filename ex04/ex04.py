import sys
sys.path.append('../ex01')
sys.path.append('../ex02')
sys.path.append('../ex03')
from ex01 import Neuron
from ex02 import OurNeuralNetwork
from ex03 import log_loss_custom
import numpy as np


students = {'Bob': (12, 15), 'Eli': (10, 9), 'Tom': (18, 18), 'Ryan': (13, 14)}
y_true = np.array([1, 0, 1, 1])

neuron_h1 = Neuron(0.05, 0.001, 0)
neuron_h2 = Neuron(0.02, 0.003, 0)
neuron_o1 = Neuron(2, 0, 0)

network = OurNeuralNetwork(neuron_h1, neuron_h2, neuron_o1)

y_pred = []
for name, (math, chem) in students.items():
    output = network.feedforward(math, chem)
    y_pred.append(output)
    print(f'{name}: {output}')

y_pred = np.array(y_pred)
print(log_loss_custom(y_true, y_pred))
