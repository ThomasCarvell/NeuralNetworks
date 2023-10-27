import numpy as np
import pickle

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

class network:

    def __init__(self,layers,activation,activationPrime):

        self.act = activation
        self.actPrime = activationPrime

        self.layers = layers

        self.biases = []
        for i in range(len(layers) - 1):
            self.biases.append(np.random.randn(layers[i+1],1))

        self.weights = []
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i+1],layers[i]))

    def saveModel(self,filename):
        with open(filename,'wb+') as f:
            f.write(pickle.dumps((self.layers,self.weights,self.biases)))

    def loadModel(self,filename):
        with open(filename,"rb") as f:
            self.layers,self.weights,self.biases = pickle.loads(f.read())
        
    def train(self,inp,target):

        biasDeriv = [np.zeros(b.shape) for b in self.biases]
        weightsDeriv = [np.zeros(w.shape) for w in self.weights]

        activation = inp
        activations = [inp]

        zs = []

        for i in range(len(self.layers)-1):
            activation = np.dot(self.weights[i],activation) + self.biases[i]
            zs.append(activation)
            activation = self.act(activation)
            activations.append(activation)

        #print(activations)

        delta = self.cost_derivative(activations[-1], target) * self.actPrime(zs[-1])

        biasDeriv[-1] = delta
        #print(delta)
        #print(np.transpose(activations[-2]))
        weightsDeriv[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.actPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            biasDeriv[-l] = delta
            weightsDeriv[-l] = np.dot(delta, activations[-l-1].T)
        
        return biasDeriv,weightsDeriv
    
    def trainBatch(self,data,learningRate):
        weightsChange = [np.zeros(w.shape) for w in self.weights]
        biasChange = [np.zeros(b.shape) for b in self.biases]

        for inp,result in data:
            biasderiv, weightsDeriv = self.train(inp,result)

            biasChange = [total+new for total,new in zip(biasChange,biasderiv)]
            weightsChange = [total+new for total,new in zip(weightsChange,weightsDeriv)]

        self.weights = [current - (learningRate/len(data)) * change for current,change in zip(self.weights,weightsChange)]
        self.biases = [current - (learningRate/len(data)) * change for current,change in zip(self.biases,biasChange)]

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    
    def forwardPropagate(self,inp):

        activation = inp

        for i in range(len(self.layers)-1):
            activation = np.dot(self.weights[i],activation) + self.biases[i]
            activation = self.act(activation)

        return activation
    
    def forwardCost(self,inp,target):
        return (self.forwardPropagate(inp) - target)**2