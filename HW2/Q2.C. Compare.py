#write your code here


import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential

import numpy as np
import matplotlib.pyplot as plt


# create inputs for sin(x)
x = np.arange(-150, 150).reshape(-1,1) / 50
y = np.sin(x)

# create MLP
model = Sequential()

model.add(Dense(30, input_shape=(1,)))
model.add(Activation('sigmoid'))
model.add(Dense(20))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='SGD')

# traing MLP for sin(x)
result = model.fit(x, y, epochs=1000, batch_size=4, verbose= 0)



X_test = np.arange(-200, 200) / 50
Y_test = np.sin(X_test)

# predict sin(x) that trained before
MLP_predictions = model.predict(X_test)
print("MLP training compeleted ")
#############################################################################


class RBF():
    def __init__(self, k=2, lr=0.01):
        self.k = k
        self.lr = lr

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)


    def fit(self, X, y, epochs=1000):

        # find centers and derivations
        self.centers = np.random.choice(X, size=self.k)
        prev = self.centers.copy()
        self.deviations = np.zeros(self.k)
        tamam = False

        while not tamam:
            distances = np.abs(X[:, np.newaxis] - self.centers[np.newaxis, :])

            closestCluster = np.argmin(distances, axis=1)

            for i in range(self.k):
                pointsForCluster = X[closestCluster == i]
                if len(pointsForCluster) > 0:
                    self.centers[i] = np.average(pointsForCluster, axis=0) 

            tamam = np.linalg.norm(self.centers - prev) < 0.00001
            prev = self.centers.copy()

        distances = np.abs(X[:, np.newaxis] - self.centers[np.newaxis, :])
        closestCluster = np.argmin(distances, axis=1)

        for i in range(self.k):
            self.deviations[i] = np.std(X[closestCluster == i])

        # training
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                a = np.array([self.G(X[i], c, s) for c, s, in zip(self.centers, self.deviations)])
                F = a.T.dot(self.w) + self.b
        
                error = -(y[i] - F)
    
                # update weights
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error

    
    # Gaussian RBF
    def G(self, x, c, s): 
        return np.exp(-1 / (2 * s**2) * (x-c)**2)


    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            a = np.array([self.G(X[i], c, s) for c, s, in zip(self.centers, self.deviations)])
            F = a.T.dot(self.w) + self.b
            predictions.append(F)
        return predictions



X = np.arange(-150, 150) / 50
Y = np.sin(X)

rbf = RBF(lr=0.01, k=2)
rbf.fit(X, Y, epochs=1000)


RBF_predictions = rbf.predict(X_test)
print("RBF training compeleted ")

plt.plot(X_test, RBF_predictions)
plt.plot(X_test, MLP_predictions)
plt.plot(X_test, Y_test)
plt.legend(["RBF prediction", "MLP prediction", "sin(x)"])
plt.xlabel('Compare')
plt.show()




