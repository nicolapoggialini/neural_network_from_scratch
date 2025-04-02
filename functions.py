import numpy as np

''' sigmoid '''

def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x)*(1 - sigmoid(x))


''' relu '''

def relu(x):
    return x*(x > 0)

def relu_der(x):
    return 1*(x > 0)


''' quadratic cost function '''

def qcf(y, pred):
    d = y - pred
    return np.sum(d**2)/(2*len(y))


def qcf_der(y, pred):
    return pred - y
