#NN from scratch
import math
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

plt.style.use('_mpl-gallery-nogrid') #theme for plt charts

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr)) #sigmoid function

def sigmoid_der(arr):
    return (1 / (1 + np.exp(-arr))) * (1 - (1 / (1 + np.exp(-arr)))) #derivative of sigmoid function

class Layer:
    def __init__(self, n, nin):
        #create arrays for weight and biases with values between -1 and 1                                 
        self.n_weights = rng.random(size=(n, nin)) * 2 - 1
        self.n_biases = rng.random(size=(n)) * 2 - 1
        self.n_grads = np.zeros((n))

    def forward(self, data):
        self.input_data = data # storing input data for calculations during backprop

        weighted_data = data * self.n_weights

        self.activations =  np.sum(weighted_data, axis=1) + self.n_biases # matrix multiplication and addition to turn weights, bias, and input into an activation

        return sigmoid(self.activations)

    def __repr__(self):
        return self.n_weights, self.n_biases

class MLP:
    def __init__(self, data, l1, l2, L):
        self.data = data
        input_n = len(data)
        #create layers
        self.l1 = Layer(l1, input_n)
        self.l2 = Layer(l2, l1)
        self.L = Layer(L, l2)

    def training_loop(self, target, training_speed, loops):
        for _ in range(loops):
            act_1 = self.l1.forward(data) #forward pass first layer
            act_2 = self.l2.forward(act_1) #forward pass second layer
            final = self.L.forward(act_2) #forward pass final layer

            print(final)

            n_losses = target - final
            total_loss = np.sum(n_losses ** 2) #calculate loss
            print(f'\nLoss: {total_loss}\n')

            #gradient is the derivative of the square function for the output neurons
            self.L.n_grads += n_losses * 2
            #grad = sum of each neuron's gradient in the previous layer, multiplied by the weight that connects the two neurons, by the sigmoid derivative
            #sum up all of the first weights of each neuron in prev layer, x by the grad of their respective neurons x by sigmoid func = grad for the first neuron, hence the np.sum(self.L.n_weights, axis=0)
            self.l2.n_grads += self.L.n_grads * np.sum(self.L.n_weights, axis=0) * sigmoid_der(self.l2.activations)
            self.l1.n_grads += self.l2.n_grads * np.sum(self.l2.n_weights, axis=0) * sigmoid_der(self.l1.activations)

            #use each neuron's grad to nudge the values of its w and b in a direction that decreases loss
            delta = self.L.n_grads * training_speed 
            self.L.n_biases += delta
            self.L.n_weights += (delta * self.L.input_data).reshape(len(self.L.n_grads), 1)#reshape to ensure each weight of a neuron is changed by the same amount
            

            delta = self.l2.n_grads * training_speed
            self.l2.n_biases += delta
            self.l2.n_weights += (delta * self.l2.input_data).reshape(len(self.l2.n_grads), 1)#reshape to ensure each weight of a neuron is changed by the same amount

            delta = self.l1.n_grads * training_speed
            self.l1.n_biases += delta
            self.l1.n_weights += (delta * self.l1.input_data).reshape(len(self.l1.n_grads), 1)#reshape to ensure each weight of a neuron is changed by the same amount

data = np.array([1, -1, 1, 1])

nn = MLP(data, 4, 4, 4)
target = np.array([1, 0, 0, 1])
nn.training_loop(target, 0.01, 1000)
