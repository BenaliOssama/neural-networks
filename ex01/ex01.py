import numpy as np

class Neuron:
    def __init__(self, weight1, weight2, bias):
        self.weights_1 = weight1
        self.weights_2 = weight2
        self.bias = bias

    def feedforward(self, x1, x2):
        # Step 1 & 2: weighted sum + bias
        total = (x1 * self.weights_1) + (x2 * self.weights_2) + self.bias
        # Step 3: sigmoid
        y = 1 / (1 + np.exp(-total))
        return y

# Test
neuron = Neuron(0, 1, 4)
print(neuron.feedforward(2, 3))
# Expected: 0.9990889488055994
