class OurNeuralNetwork:
    def __init__(self, neuron_h1, neuron_h2, neuron_o1):
        self.h1 = neuron_h1
        self.h2 = neuron_h2
        self.o1 = neuron_o1

    def feedforward(self, x1, x2):
        out_h1 = self.h1.feedforward(x1, x2)
        out_h2 = self.h2.feedforward(x1, x2)
        # o1 takes the hidden layer outputs as its inputs
        y = self.o1.feedforward(out_h1, out_h2)
        return y

# Test (keep your Neuron class above this)
neuron_h1 = Neuron(1, 2, -1)
neuron_h2 = Neuron(0.5, 1, 0)
neuron_o1 = Neuron(2, 0, 1)

network = OurNeuralNetwork(neuron_h1, neuron_h2, neuron_o1)
print(network.feedforward(2, 3))
# Expected: 0.9524917424084265
