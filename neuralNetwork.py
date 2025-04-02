import numpy as np
from methods import out_matrix


class NeuralNetwork:
    
    def __init__(self, loss_fun, loss_der):
        
        self.X = None
        self.y = None
        self.n = 0
        self.total_cost = 0
        self.layers = []
        self.loss_fun = loss_fun
        self.loss_der = loss_der

    
    def update_input(self, new_X, new_y):

        self.X = new_X
        self.y = new_y
        self.n = len(new_X)

    
    def add_layer(self, layer):
        
        self.layers.append(layer)
    
        
    def feedforward(self):
        
         input_values = self.X
         for layer in self.layers:
             layer.z = np.dot(input_values, layer.weights) + layer.bias
             layer.a = layer.act_fun(layer.z)
             input_values = layer.a
             
        
    def backpropagation(self, learning_rate):
        
        y_matrix = out_matrix(self.y, self.layers[-1].size)
        pred = self.layers[-1].a
        # dC_da
        error = self.loss_der(y_matrix, pred)
        # total cost of the network
        total_cost = self.loss_fun(y_matrix, pred)
        self.total_cost = self.total_cost + total_cost
        
        for i in reversed(range(len(self.layers))):
            
             delta = error*self.layers[i].der(self.layers[i].z)
             # backpropagate the error
             error = np.dot(delta, self.layers[i].weights.T)
             
             if i == 0:
                 previous_act = self.X.T
             else:
                previous_act = self.layers[i - 1].a.T
             # gradient descend
             dC_dw = np.dot(previous_act, delta)/self.n
             dC_db = np.sum(delta, axis = 0)/self.n
             self.layers[i].weights = self.layers[i].weights - learning_rate*dC_dw
             self.layers[i].bias = self.layers[i].bias - learning_rate*dC_db
           
    
    
    def predict_activations(self, x_test, y_test):
        
        self.update_input(x_test, y_test)
        self.feedforward()
        output_layer_activations = self.layers[-1].a
        return output_layer_activations


    def predicted_labels(self, activations):
        
        pred_labels = np.argmax(activations, axis = 1)
        return pred_labels



    def get_tot_cost(self):
        
        return self.total_cost


    def init_tot_cost(self):
        
        self.total_cost = 0
            

    def mini_batch(self, X, y, size, epochs, learning_rate):
        
        n = len(X)
        index = np.arange(n)
        np.random.shuffle(index)
        start = 0
        end = size
        n_batch = n//size
        for e in range(epochs):
            for k in range(n_batch):
                ind = index[start:end]
                batch = X[ind]
                y_batch = y[ind]
                self.update_input(batch, y_batch)
                self.feedforward()
                self.backpropagation(learning_rate)
                start = start + size
                end = end + size
            total_cost = self.get_tot_cost()
            print('Epoch: ' + str(e + 1) + '   loss: ' + str(total_cost/n_batch))
            self.init_tot_cost()
            np.random.shuffle(index)
            start = 0
            end = size
