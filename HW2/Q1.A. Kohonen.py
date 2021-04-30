#write your code here

import numpy as np
import matplotlib.pyplot as plt


# adjust inputs here :
neurons = 1600
size = int(np.sqrt(neurons))
learning_rate = 0.1
sigma = 5
epochs = 100

########################################################
def find_dists(a, index):
    i,j = np.indices(a.shape, sparse=True)
    return (i-index[0])**2 + (j-index[1])**2

def convert_reshape(a):
    a0 = (int) (a/size)
    a1 = a % size
    return a0, a1

# initialize random input and weights
weights = np.random.rand(neurons, 3) 
inputs = np.random.rand(neurons,3)

# show random input rgb data
plt.imshow(inputs.reshape(size, size,3))
plt.show()

for e in range(epochs):
    for x in range(neurons):
        # find best neuron 
        arg_bmu = np.argmin(np.sum(np.square(np.subtract(weights, inputs[x])), axis=1))
        
        # find dist vector from all neurons to bmu
        dist = find_dists(np.zeros((size,size)), tuple(convert_reshape(arg_bmu))).reshape(-1,1)

        # update weights
        weights += np.multiply(inputs[x] - weights, np.exp((-dist)/(2*(sigma)**2)))*learning_rate

# show results (by weights)
plt.imshow(weights.reshape(size,size,3))
plt.show()





