#NN from scratch
import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x)) #sigmoid function
def sigmoid_der(x):
    return (1 / (1 + math.exp(-x))) * (1 - (1 / (1 + math.exp(-x)))) #derivative of sigmoid function

class Neuron:
    def __init__(self, nin):
        self.w = [random.uniform(-1,1) for _ in range(nin)] #random array of weights for each input (signal strength)
        self.b = random.uniform(-1,1)                       #random array of biases for each input (trigger-happiness of neuron)
        self.grad = 0
        self.grad = 0
        self.activation = None                              #gradient set to 0, will be used to nudge values later

    def forward(self, data):
        self.activation = sigmoid((sum(w * d for w, d in zip(self.w, data))/len(data)) + self.b) #multiply inputs with weights, then add bias, sigmoid function to set value between 0 and 1
        return self.activation

class Layer:
    def __init__(self, n, nin):                                 
        self.neurons = [Neuron(nin) for _ in range(n)] #create n neurons to make a layer

    def forward(self, data):
        return [neuron.forward(data) for neuron in self.neurons]
        
    def __repr__(self):
        return f'{self.neurons}'

class MLP:
    def __init__(self, data, l1, l2, L):
        self.data = data
        input_n = len(data)
        #create layers
        self.l1 = Layer(l1, input_n)
        self.l2 = Layer(l2, l1)
        self.L = Layer(L, l2)

    def training_loop(self, target, training_speed, loops):
        for i in range(loops):
            act_1 = self.l1.forward(data) #forward pass first layer
            act_2 = self.l2.forward(act_1) #forward pass second layer
            final = self.L.forward(act_2) #forward pass final layer

            round_final = [round(n, 5) for n in final]
            print(round_final)
          
            loss = sum((predict - targ)**2 for predict, targ in zip(final, target)) #calculate loss
            print(f'\nLoss: {loss}\n')
        
            #give each neuron a gradient showing how much that neuron affects the final outcome
            for neuron, targ in zip(self.L.neurons, target):
                #gradient is the derivative of the square function for the output neurons
                neuron.grad += 2*(neuron.activation - targ)

            for i, neuron in enumerate(self.l2.neurons):
                # grad = sum of each neuron's gradient in the previous layer, multiplied by the weight that connects the two neurons, by the sigmoid derivative
                neuron.grad += sum(g * w for g, w in zip((n.grad for n in self.L.neurons), (n.w[i] for n in self.L.neurons))) * sigmoid_der(neuron.activation) 

            for i, neuron in enumerate(self.l1.neurons):
                # grad = sum of each neuron's gradient in the previous layer, multiplied by the weight that connects the two neurons, by the sigmoid derivative
                neuron.grad += sum(g * w for g, w in zip((n.grad for n in self.l2.neurons), (n.w[i] for n in self.l2.neurons))) * sigmoid_der(neuron.activation)

            #use each neurons grad to nudge the values of its w and b in a direction that decreases loss
            for neuron in self.L.neurons:
                for i, weight in enumerate(neuron.w):
                    weight += -training_speed * neuron.grad * self.l2.neurons[i].activation
                neuron.b += -training_speed * neuron.grad
            for neuron in self.l1.neurons:
                for i, weight in enumerate(neuron.w):
                    weight += -training_speed * neuron.grad * self.l1.neurons[i].activation
                neuron.b += -training_speed * neuron.grad
            for neuron in self.l2.neurons:
                for i, weight in enumerate(neuron.w):
                    weight += -training_speed * neuron.grad * self.data[i]
                neuron.b += -training_speed * neuron.grad

            print()

    def __str__(self):
        return f'{self.data}: input data\n'
    
data = [3, 6, -1, 4]

nn = MLP(data, 4, 4, 4)
print('\n', nn, '\n')
target = [1, 0, 0.5, 1]
print('\n', nn.training_loop(target, 0.1, 20))



