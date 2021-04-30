# Q5_graded
# Do not change the above line.

# This cell is for your imports

import numpy as np
import matplotlib.pyplot as plt


# Q5_graded
# Do not change the above line.
class MLP():
    def __init__(self, input_layer=2, hidden_layer1=2, output_layer=2, etha=0.5):
        # initialize properties
        self.l1 = input_layer
        self.l2 = hidden_layer1
        self.l3 = output_layer

        self.w1 = np.random.rand(hidden_layer1, input_layer) 
        self.b1 = np.zeros((hidden_layer1, 1))
        self.w2 = np.random.rand(output_layer, hidden_layer1) 
        self.b2 = np.zeros((output_layer, 1))

        self.input = []
        self.Y = []
        
        self.net1 = []
        self.out1 = []
        self.net2 = []
        self.out2 = []

        self.err = np.zeros((output_layer,1))
        self.etha = etha

    def softmax(self, inputs):
        inputs = inputs - np.max(inputs, axis=0, keepdims=True)
        n = np.exp(inputs)
        result = n / np.sum(n, axis=0, keepdims=True)
        return result  
       
    def forward(self, input):
        # save inputs this class properties
        self.input = np.copy(input)
        
        # calculate net for hidden layer
        self.net1 = np.dot(self.w1, input) + self.b1

        ## activataion function for calculate out1
        self.out1 = 1/(1+np.exp(-self.net1)) # sigmoid
        # self.out1 = np.maximum(0,self.net1) # relu

        # calculate net for output layer
        self.net2 = np.dot(self.w2, self.out1) + self.b2

        ## activataion function for calculate out2
        self.out2 = self.softmax(self.net2) # softmax
        # self.out2 = np.maximum(0,self.net2) # relu
        # self.out2 = 1/(1+np.exp(-self.net2)) # sigmoid

        return self.out2


    def Backward(self, target):
        # save target to this class 
        self.Y = target

        # calculate error 
        self.err = np.multiply(np.subtract(self.Y, self.out2), np.subtract(self.Y, self.out2)) * 0.5
       
        # calculate gradient for first layer back
        t1 = np.subtract(self.out2, self.Y)
        t2 = np.multiply(self.out2, np.subtract(1, self.out2))
        a = np.multiply(t1, t2)
        wtf = np.dot(a,self.out1.T)
        changes = wtf * self.etha

        # calculate new w2
        w2_n = np.subtract(self.w2, changes)

        # update bios first layer back
        self.b2 = np.subtract(self.b2, a * self.etha)


        # calculate gradient for second layer back
        x1 = np.dot(self.w2.T, a)
        x2 = np.multiply(self.out1, np.subtract(1, self.out1))
        a2 = np.multiply(x1, x2)
        wtf2 = np.dot(a2, self.input.T) 
        changes2 = wtf2 * self.etha

        # calculate new w1 and update it
        self.w1 = np.subtract(self.w1, changes2)

        # update w2 
        self.w2 = w2_n

        # update bios second layer back
        self.b1 = np.subtract(self.b1, a2 * self.etha)


    def train(self, input, label):
        # fit the data
        acc = self.forward(input)
        self.Backward(label)

        # calculate loss 
        loss = np.sum(self.err)

        return np.argmax(acc), loss


    def predict(self, input):
        out = self.forward(input)
        return np.argmax(out)

if __name__=='__main__':

    # open and read mnist database
    mnist_file = open("./sample_data/mnist_train_small.csv", "r")
    lines = mnist_file.readlines()
    mnist_test_file = open("/content/sample_data/mnist_test.csv", "r")
    lines_test = mnist_test_file.readlines()
 
    # create MLP object
    nn = MLP(784,85,10,0.3)    
    r = 0
    for x in range(20):
        r += 1
        print(r,end=" --> ")
        i = 0   
        total_loss = 0
        t=0
        f=0 
        for line in lines:
            # parse and normalize input 
            y_train = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
            # y_train = np.zeros((10,1))
            y_train = y_train.reshape(10,1).astype('float32')
            toks = line.split(",")
            toks = [ int(x) for x in toks]
            y_train[toks[0]] = [0.99]
            x_train = np.array(toks[1:]).reshape(784, 1).astype('float32')
            x_train /= 255
            
            # train mnist
            ans, loss = nn.train(x_train, y_train)
            total_loss += loss
            if (ans == toks[0]):
                t+=1
            else:
                f+=1
            if(i % 1000 == 0):
                # print progress
                print("**",end='')
                
            i += 1
            
        print("  loss:", total_loss/20000, "  acc:", t/(t+f))
        
    # predict tests data
    t=0
    f=0 
    for line in lines_test:
        toks = line.split(",")
        toks = [ int(x) for x in toks]
        x_test = np.array(toks[1:]).reshape(784, 1).astype('float32')  
        x_test /= 255
        x = nn.predict(x_test)
        if ( x == toks[0]):
            t+=1
        else:
            f+=1
    print("tests acc= ", t/(t+f))
    
   



# Q5_graded
# Do not change the above line.

# This cell is for your codes.

