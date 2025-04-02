import numpy as np
import matplotlib.pyplot as plt 



def out_matrix(y, lastlayer_dim):
    
    if lastlayer_dim == 1:
        return y.reshape(len(y), 1)
    else:
        m = np.zeros((len(y), lastlayer_dim))
        for i in range(len(m)):
            m[i][y[i]] = 1
        return m
    
    

def print_accuracy(y, pred):
    accuracy = np.sum(pred == y)/len(y)
    print('Accuracy on test set: ', accuracy)
    
    
     
def pred_dict(pred, y):
    
    d = {}
    for i in range(len(y)):
        if y[i] in d:
            d[y[i]][0] = d[y[i]][0] + 1 # first value: number of cases
            if y[i] == pred[i]:
                d[y[i]][1] = d[y[i]][1] + 1 # second value: number of true predictions
        elif y[i] == pred[i]:
            d[y[i]] = [1,1]
        else:
            d[y[i]] = [1,0]
    return d




def print_recall_precision(d_rec, d_prec, class_names):
    
    for e in class_names:
        rec = d_rec[e][1]/d_rec[e][0]
        if e in d_prec:
            prec = d_prec[e][1]/d_prec[e][0]
        else:
            prec = 0.0
        print(class_names[e] + ': recall = ' + str(rec) + '     precision = ' + str(prec))
        




def plot_image(i, predicted_labels, true_labels, images, class_names):

    pred_label = predicted_labels[i]
    true_label = true_labels[i]
    image = images[i]
    plt.xticks([])
    plt.yticks([])  
    plt.imshow(image, cmap = 'gray')    
    if pred_label == true_label:
      color = 'green'
    else:
      color = 'red' 
    plt.title(class_names[pred_label] + ' (' + class_names[true_label] + ')', color = color)




def plot_activations(i, pred_activations, pred_labels, true_labels):

    pred_a = pred_activations[i]
    pred_label = pred_labels[i]
    true_label =  true_labels[i]
    plt.xticks(range(10))
    plt.yticks([])
    act_plot = plt.bar(range(10), pred_a)
    plt.ylim([0, 1])    
    act_plot[pred_label].set_color('red')
    act_plot[true_label].set_color('green')
