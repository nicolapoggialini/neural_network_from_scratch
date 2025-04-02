import numpy as np

class layer:
    
    def __init__(self, input_dim, output_dim, act_fun, der):
        
        self.size = output_dim
        self.weights = np.random.randn(input_dim, output_dim)*np.sqrt(2/(input_dim + output_dim))
        self.bias = np.zeros((1, output_dim))
        self.z = None
        self.a = None
        self.act_fun = act_fun
        self.der = der
        