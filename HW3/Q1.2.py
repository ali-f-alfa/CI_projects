# Q1.2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q1.2_graded
# Do not change the above line.

class Hopfield:
    def __init__(self, dim):
        # set dimension of network (number of nuerons)
        self.dim = dim
        # initialize weight matrix with zeros
        self.weights = np.zeros((dim, dim)).astype('int')

    def Save(self, input_list):
        # convert input to numpy array
        input_list = np.array(input_list)

        # create reverse of input
        input_list_r = input_list.reshape(-1,1)

        # calculate new weights that our input generates
        new_weights = input_list_r * input_list

        # remove diagonal of matrix
        np.fill_diagonal(new_weights, 0)

        # update our final weights matrix
        self.weights += new_weights
        print("updated weights by input: ", input_list, "is :\n", self.weights) 
 

    def CheckInput(self, input):
        # convert input to numpy array
        input = np.array(input)

        # calculate dot of input and weights matrix
        input_mult = np.dot(self.weights, input)

        # find sign of calculated data before
        input_sign = np.sign(input_mult)

        if ((input_sign == input).all()):
            print(input,"is STABLE")
        else:
            print(input, "is UNSTABLE, the nearest saved data is: ", input_sign)



 
if __name__=='__main__':
    hop = Hopfield(6)
    hop.Save([1,1,1,-1,-1,-1])
    hop.Save([1,-1,1,-1,1,-1])
    hop.CheckInput([1,1,1,-1,-1,-1])
    hop.CheckInput([1,-1,1,-1,1,-1])
    hop.CheckInput([-1,1,1,-1,-1,-1])


# This cell is for your codes.

