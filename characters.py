import numpy as np
from neuralNets import *
from PIL import Image
import matplotlib.pyplot as plt
import os, random


net = network([28*28,15,10],sigmoid,sigmoid_prime)
#net.loadModel("trainedDigits20000x2+50000x3.net")

def randomNum(num,training = True):
    if training:
        train = "training"
    else:
        train = "testing"
    
    filename =  random.choice(os.listdir(f"mnist_png\\{train}\\{num}"))
    img = Image.open(f"mnist_png\\{train}\\{num}\\{filename}")
    data = np.asarray(img) / 255
    data.resize((28*28,1))

    return data

def evaluateCost(network, iterations = 10):
    cost = 0
    
    for i in range(iterations):
        num = np.random.randint(0,10)
        target = np.zeros(10)
        target[num] = 1

        data = randomNum(num,False)

        result = network.forwardPropagate(data).T[0]

        cost += sum((target-result)**2)

    return cost/iterations

def trainBatch(network,learningRate = 5, iterations = 10):

    batch = []

    for i in range(iterations):
        n = np.random.randint(0,10)
        target = np.zeros((10,1))
        target[n][0] = 1
        
        batch.append((randomNum(n),target))

    network.trainBatch(batch,learningRate)

totalBatches = 10000

batchNo = []
costs = []

for i in range(totalBatches):
    if i%200 == 0:
        print(f"Batch {i}: {round(i/totalBatches * 100,1)}%")
        batchNo.append(i)
        costs.append(evaluateCost(net,50))

    trainBatch(net,0.75,10)

net.saveModel("NeuralNet.net")

plt.plot(batchNo,costs)

plt.pause(1000)

# while True:
#     input()
#     n = np.random.randint(0,9)
#     d = randomNum(n,False)
#     p = net.forwardPropagate(d)

#     fig, ax = plt.subplots(2)
#     ax[0].imshow(np.reshape(d,(28,28)))
#     ax[1].imshow(p)

#     plt.show()
