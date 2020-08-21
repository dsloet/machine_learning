import numpy as np
from LinearRegressionModel import LinearRegressionModel

# Generate 'random' data

#np.random.seed(0)
#X = 2.5 * np.random.randn(100,2) + 1.5   

#res = 0.5 * np.random.randn(100)       

#y = 2 + 0.3 * X + res                  

#X = X.reshape(100,2)
#y = y.reshape(100,1)

#lr = LinearRegressionModel(epochs = 15000)
#lr.train(X,y)
#y_hat = lr.predict(X)


#import matplotlib.pyplot as plt

#plt.subplot()
#plt.scatter(X, y, c='r')
#plt.plot(X, y_hat, c='b')

from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)
ones = np.ones([X.shape[0], 1])

ones = np.ones([X.shape[0], 1])
X = np.concatenate((ones, X), axis=1)
theta = np.zeros([1,X.shape[1]])

temp0 = np.power(X.dot(theta.T)-y,2)
np.sum(temp0)/(2 * len(X))

def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

def gradientDescent(X,y,theta,iters = 100,alpha = 0.01):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost    

g,cost = gradientDescent(X,y,theta)
print(g)

finalCost = computeCost(X,y,g)
print(finalCost)