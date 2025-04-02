import matplotlib.pyplot as plt
import keras
from layer import layer
from neuralNetwork import NeuralNetwork
from functions import sigmoid, sigmoid_der, relu, relu_der, qcf, qcf_der
from methods import print_accuracy, pred_dict, print_recall_precision, plot_image, plot_activations

fashion_mnist = keras.datasets.fashion_mnist


# load dataset
(train_images, train_y), (test_images, test_y) = fashion_mnist.load_data()

# reshape data
trainX = train_images.reshape(len(train_images), train_images.shape[1]*train_images.shape[2])

testX = test_images.reshape(len(test_images), test_images.shape[1]*test_images.shape[2])

# BW
trainX = trainX / 255.0
testX = testX / 255.0


class_names = {0: 'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat',
                5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}



nn = NeuralNetwork(qcf, qcf_der)
layer1 = layer(trainX.shape[1], 50, relu, relu_der)
nn.add_layer(layer1)
layer2 = layer(50, 30, relu, relu_der)
nn.add_layer(layer2)
layer3 = layer(30, 10, sigmoid, sigmoid_der)
nn.add_layer(layer3)

nn.mini_batch(trainX, train_y, 32, 10, 0.1)

pred_a = nn.predict_activations(testX, test_y)
pred_labels = nn.predicted_labels(pred_a)

# accuracy
print(' --------------------- ')
print_accuracy(test_y, pred_labels)
print(' --------------------- ')

# recall / precision
d_rec = pred_dict(pred_labels, test_y)
d_prec = pred_dict(test_y, pred_labels)
print_recall_precision(d_rec, d_prec, class_names)

# print examples
start = 0
end = 15
for i in range(start, end):
    plt.subplot(1,2,1)
    plot_image(i, pred_labels, test_y, test_images, class_names)
    plt.subplot(1,2,2)
    plot_activations(i, pred_a, pred_labels, test_y)
    plt.show()
