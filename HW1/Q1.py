# Q1_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q1_graded
# Do not change the above line.

class NOR:
    def __init__(self, learning_rate, n):
        self.input_dim = n 
        self.weights = np.random.rand(self.input_dim)
        self.learning_rate = learning_rate
        self.max_change = 100

    def learn(self,learning_input, learning_answer):
        y = np.dot(learning_input, self.weights)
        error = learning_answer - y
        # print("error: ",error)
        # print("weight before: ",self.weights)


        # Update weight for each edge & Find Largest weight change
        change = -1
        for i in range(self.input_dim):
            new_weight = self.weights[i] + (self.learning_rate*error*learning_input[i])
            if (abs(new_weight - self.weights[i]) > change ):
                change = abs(new_weight - self.weights[i])
            self.weights[i] = new_weight
        
        # decrease learning rate to avoid infinite loop
        self.learning_rate = self.learning_rate *0.999

        
        # print("learning rate: ",self.learning_rate)
        # print("max_change : ",self.max_change)
        self.max_change = change

        # print("weight after: ",self.weights)


# This cell is for your codes.

# Q1_graded
# Do not change the above line.

nor = NOR(0.1, 3)
limit = 0.000000001
x = 0
while(nor.max_change > limit):
    x += 1
    # print("************************")
    # print("[-1, -1] ==> 1")
    nor.learn(np.array([1,-1,-1]), 1)
    # print("************************")
    # print("[-1, 1] ==> -1")
    nor.learn(np.array([1,-1,1]), -1)
    # print("************************")
    # print("[1, -1] ==> -1")
    nor.learn(np.array([1,1,-1]), -1)
    # print("************************")
    # print("[1, 1] ==> -1")
    nor.learn(np.array([1,1,1]), -1)

print("iterations: ", x)
print("final weights: ",nor.weights)
# This cell is for your codes.

