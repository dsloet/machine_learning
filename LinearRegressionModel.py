import numpy as np

class LinearRegressionModel():
    """Linear regression model"""

    def __init__(self, epochs = 15000, alpha = 0.0005):
        self.bias = 0
        self.weights = 0
        self.epochs = epochs
        self.alpha = alpha
        self.dW = 0.0
        self.dB = 0.0

    def avg_loss(self, x, y):
        N = len(x)
        self.total_error = 0
        
        for i in range(N):
            self.total_error += (y[i] - (self.weights*x[i] + self.bias))**2
        return(self.total_error)
       
    def update_derivates(self, x, y):
        
        for i in range(len(x)):
            self.dW += -2*x[i]*(y[i] - (self.weights * x[i] + self.bias))
            self.dB += -2 * (y[i] - (self.weights * x[i] + self.bias))

        self.weights = self.weights - 1/len(x) * self.dW * self.alpha
        self.bias = self.bias - 1/len(x) * self.dB * self.alpha

        self.dW = 0.0
        self.dB = 0.0

        return(self.weights, self.bias)
       
        
    def train(self, x, y):

        for i in range(self.epochs):
        
            self.weights, self.bias = self.update_derivates(x, y)
    
            if i % 500 == 0:
                self.total_error = self.avg_loss(x, y)
                print("\n\nepoch = ", i)
                print("weights = ", self.weights, "\nbias = ", self.bias, "\ntotal error = ", self.total_error)


    def predict(self, x):
        return(self.weights * x + self.bias)
    
    
