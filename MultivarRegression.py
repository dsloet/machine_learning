import numpy as np

class MultivarReg():
    def __init__(self, epochs = 100, alpha = 0.001):
        self.epochs = epochs
        self.alpha = alpha
        #self.theta = 0.0

    def computeCost(self, X, y):
        temp0 = np.power(((X @ self.theta.T)-y),2)
        cost = np.sum(temp0)/(2 * len(X))
        return(cost)
    
    def gradientDescent(self, X, y):
        self.cost = np.zeros(self.epochs)

        for i in range(self.epochs):
            self.theta = self.theta - (self.alpha/len(X)) * np.sum(X * (X @ self.theta.T - y), axis=0)
            self.cost[i] = self.computeCost(X, y)
            if self.cost[i] == np.nan:
                print("Infinity")
                break
            print(self.cost[i])
            
        return(self.theta, self.cost)

    def fit(self, X, y, meanscale = True):
        if meanscale:
            X, y = self.meanscale(X,y)
            #self.theta = np.zeros([1, X.shape[1]]) # these are the trained parameters
            self.theta = np.random.randn(1, X.shape[1]) # random initialisation of theta
            self.theta, self.cost = self.gradientDescent(X,y)
        else:
            self.theta = np.zeros([1, X.shape[1]]) 
            self.theta, self.cost = self.gradientDescent(X,y)

        #return(self.theta, self.cost)
    def meanscale(self, X, y):
        X = (X - X.mean())/ X.std()
        y = (y - y.mean())/ y.std()
        return X, y

    def predict(self, X):
        return -1 * (X @ self.theta.T)

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
ones = np.ones([X.shape[0], 1]) # these are the biases

# add bias to X
X = np.concatenate((ones, X), axis=1)
y = y.reshape(y.shape[0], 1)

mvr = MultivarReg(epochs = 30000, alpha=0.001)
mvr.fit(X,y, meanscale=True)

#(-1 * (X @ mvr.theta.T))