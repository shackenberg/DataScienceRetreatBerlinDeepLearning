"""generate_gradient.py
~~~~~~~~~~~~~~~~~~~~~~
Adapted from: https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/fig/generate_gradient.py
Copyright Michael Nielsen

For usage with ipython notebook:
import matplotlib
%matplotlib
import generate_gradient
# network with two hidden layers
generate_gradient.main([784, 30, 30, 10])

"""

#### Libraries
# Standard library
import json
import math
import random
import shutil

# My library
import mnist_loader
import network

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np

def main(layers=None):
    # Load the data
    full_td, _, _ = mnist_loader.load_data_wrapper()
    td = full_td[:1000] # Just use the first 1000 items of training data
    epochs = 500 # Number of epochs to train for

    if layers is None:
      layers = [784, 30, 30, 10]
    net = network.Network(layers)
    initial_norms(td, net)
    abbreviated_gradient = [
        ag[:6] for ag in get_average_gradient(net, td)[:-1]]
    training(td, net, epochs, "norms_during_training_2_layers.json")
    plot_training(
        epochs, "norms_during_training_2_layers.json", 2)
0

def initial_norms(training_data, net):
    average_gradient = get_average_gradient(net, training_data)
    norms = [list_norm(avg) for avg in average_gradient[:-1]]
    print "Average gradient for the hidden layers: "+str(norms)

def training(training_data, net, epochs, filename):
    norms = []
    for j in range(epochs):
        average_gradient = get_average_gradient(net, training_data)
        norms.append([list_norm(avg) for avg in average_gradient[:-1]])
        print "Epoch: %s" % j
        net.SGD(training_data, 1, 1000, 0.1)
    f = open(filename, "w")
    json.dump(norms, f)
    f.close()

def plot_training(epochs, filename, num_layers):
    f = open(filename, "r")
    norms = json.load(f)
    f.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["#2A6EA6", "#FFA933", "#FF5555", "#55FF55", "#5555FF"]
    for j in range(num_layers):
        ax.plot(np.arange(epochs),
                [n[j] for n in norms],
                color=colors[j],
                label="Hidden layer %s" % (j+1,))
    ax.set_xlim([0, epochs])
    ax.grid(True)
    ax.set_xlabel('Number of epochs of training')
    ax.set_title('Speed of learning: %s hidden layers' % num_layers)
    ax.set_yscale('log')
    plt.legend(loc="upper right")
    plt.show()

def get_average_gradient(net, training_data):
    nabla_b_results = [net.backprop(x, y)[0] for x, y in training_data]
    gradient = list_sum(nabla_b_results)
    return [(np.reshape(g, len(g))/len(training_data)).tolist()
            for g in gradient]

def zip_sum(a, b):
    return [x+y for (x, y) in zip(a, b)]

def list_sum(l):
    return reduce(zip_sum, l)

def list_norm(l):
    return math.sqrt(sum([x*x for x in l]))
