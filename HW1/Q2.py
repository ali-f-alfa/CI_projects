# Q2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q2_graded
# Do not change the above line.

class Perceptron:
    def __init__(self, learning_rate, n):
        self.input_dim = n 
        self.weights = np.random.rand(self.input_dim)
        self.learning_rate = learning_rate
        self.max_change = 100
        self.mis = 0

    def Calculate_Weights_Average(self):
        average = 0
        for i in range(self.input_dim):
            average += self.weights[i]
        average /= self.input_dim
        return average
    
    # Activation function
    def Is_active(self, y, label):
        if (((y <= 0) and (label > 0)) or ((y > 0) and (label < 0))):
            return True
        return False

    def learn(self,learning_input, learning_answer):

        # calculate dots of input training data and current weights
        y = np.dot(learning_input, self.weights)
        
        # Check answer is true or not & calculate mistakes
       
        if (not self.Is_active(y, learning_answer)):
            self.mis += 1
        
        # Update weight for each edge & Find Largest weight change
        error = learning_answer - y
        change = -1
        for i in range(self.input_dim):
            new_weight = self.weights[i] + (self.learning_rate*error*learning_input[i])
            if (abs(new_weight - self.weights[i]) > change ):
                change = abs(new_weight - self.weights[i])
            self.weights[i] = new_weight

        self.max_change = change
        
        
        # Calculate Average of Weights to avoid large numbers
        average = self.Calculate_Weights_Average()
        if ( average != 0 ):
            self.weights /= average/1000
        

        # decrease learning rate to avoid infinite loop
        self.learning_rate = self.learning_rate *0.9999
      

# This cell is for your codes.

# Q2_graded
# Do not change the above line.

# open and read data file
data_file = "/content/data.txt"
data = open(data_file, "r")
FileContentLines = data.readlines()

# initialize our perceptron
BC = Perceptron(0.2, 3)

# parse data and fill lists for later uses
points_zero_x = []
points_zero_y = []
points_one_x = []
points_one_y = []

all_points = []
labels = []

mistakes = [] 

for line in FileContentLines:
    toks = line.split(',')
    if(float(toks[2]) == 0):
        toks[2] = "-1"
        points_zero_x.append(float(toks[0]))
        points_zero_y.append(float(toks[1]))
    else:
        points_one_x.append(float(toks[0]))
        points_one_y.append(float(toks[1]))

    all_points.append([1, float(toks[0]), float(toks[1])])
    labels.append(float(toks[2]))


# start training with input data until largest weight change becomes less than limit & 
# count number of mistakes in each iteration
limit = 0.000001
iter = 0
while (BC.max_change > limit):
    for i in range(len(all_points)):
        BC.learn(all_points[i], labels[i])
    iter += 1
    mistakes.append(BC.mis)
    BC.mis = 0
    # print(BC.weights)


# print number of iterations and final weights
print("iterations: ", iter)
print("final weights: ",BC.weights)


# plotting the Points and final Separator Line
plt.scatter(points_zero_x, points_zero_y,marker='.')
plt.scatter(points_one_x, points_one_y,marker='*')
x = np.linspace(-210, -60, 400)
t = BC.weights
Y = -1*(((t[1]/t[2])*x) + (t[0]/t[2]))
plt.plot(x, Y,linestyle='solid')
plt.title('Result of classification by perceptron')
plt.show()

print()
# plotting mistakes in each iteration 
plt.plot(mistakes)
plt.xlabel('Iterations')
plt.ylabel('Mistakes')
plt.title('Number of Mistakes in each iteration')
plt.show()

# This cell is for your codes.

