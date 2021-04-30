import numpy as np
import matplotlib.pyplot as plt


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
y = np.sin(X)

rbf = RBF(lr=0.01, k=2)
rbf.fit(X, y, epochs=1000)

predictions = rbf.predict(X)

plt.plot(X, predictions)
plt.plot(X, y)
plt.legend(["predict", "sinx"])
plt.xlabel('RBF')
plt.show()



