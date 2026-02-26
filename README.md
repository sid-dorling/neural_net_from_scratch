# neural_net_from_scratch
Neural network built completely from scratch

Includes an input and output layer with 2 hidden layers. The number of neurons in each layer is variable. Each Neuron receives a "grad" during backpropogation, which shouws the neuron's impact on the loss, and is used to adjust the neuron in a way that decreases the loss
Each layer uses sigmoid function for binary classification and non-linearity, outputs range from 0.0 to 1.0 due to the output layer also having a sigmoid function.
