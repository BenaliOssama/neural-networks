# Neural Networks

## Overview

Implementation of a neural network from scratch using only NumPy. Covers the core mechanics of how a neural network works: neurons, forward propagation, loss functions, and the difference between classification and regression output layers.

No training is performed — weights are provided manually. The goal is to understand how data flows through a network before tackling backpropagation.

## Project Structure

```
.
├── ex01/
│   └── ex01.py        # Single neuron
├── ex02/
│   └── ex02.py        # 3-neuron network
├── ex03/
│   └── ex03.py        # Log loss
├── ex04/
│   └── ex04.py        # Forward propagation on real data
└── ex05/
    └── ex05.py        # Regression output layer
```

## Exercises

### Ex01 — The Neuron
A single artificial neuron computes:
```
y = sigmoid(w1*x1 + w2*x2 + b)
```
This is identical to logistic regression on two inputs. The sigmoid squashes any number into (0, 1).

### Ex02 — Neural Network
Three neurons connected together. The outputs of h1 and h2 become the inputs of o1. No new math — just chaining the same operation across layers.

### Ex03 — Log Loss
Loss function for classification. Penalizes confident wrong predictions harder than MSE would.
```
-1/n * sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
```

### Ex04 — Forward Propagation
Runs the network on real student data (math and chemistry grades) and measures prediction error using log loss.

### Ex05 — Regression
To predict a real-valued output (a grade, not a probability), the output neuron replaces sigmoid with the identity function `f(x) = x`. Hidden neurons are unchanged. Error is measured with MSE.

## Key Concepts

- **Weights and bias**: the parameters of each neuron. In a real network these are learned through backpropagation — here they are given manually.
- **Activation function**: sigmoid for classification (squashes to 0-1), identity for regression (lets the raw value through).
- **Forward propagation**: passing input data through the network layer by layer to get a prediction.
- **Loss function**: measures how wrong the predictions are. Log loss for classification, MSE for regression.
- **Black box**: the weights are deterministic and explainable in principle, but the network gives no human-readable reason for its decisions.
- **Universal approximation**: a neural network with enough neurons can approximate any function — just by adjusting weights.

## Setup

```bash
python3 -m venv ex00
source ex00/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python3 ex01/ex01.py
python3 ex02/ex02.py
# etc.
```
