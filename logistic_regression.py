import numpy as np

class LogisticRegression:
    """Simple log reg class"""

    def __init__(self):
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_function(self, X, y):
        yhat = self.sigmoid(np.dot(X, self.weights))
        pred_1 = y * np.log(yhat)  # if yhat=0, this part will be zero
        pred_0 = (1 - y) * (
            np.log(1 - yhat)
        )  # if yhat=1 this will be zero (because of the 1-y bit)
        loss = -sum(pred_1 + pred_0) / len(X)
        return loss

    def fit(self, X, y, learning_rate=0.025, epochs=1000):
        self.weights = np.random.normal(size=X.shape[1])
        self.loss = []
        lr = learning_rate
        for _ in range(epochs):
            # gradient descent
            yhat = self.sigmoid(np.dot(X, self.weights))
            self.weights -= lr * np.dot(X.T, (yhat - y)) / len(X)
            self.loss.append(self.loss_function(X, y))

    def predict_proba(self, X):
        if not isinstance(self.weights, np.ndarray):
            print("Please fit the model first")
        pred_prob = self.sigmoid(np.dot(X, self.weights))
        return pred_prob

    def predict(self, X, threshold=0.5):
        if not isinstance(self.weights, np.ndarray):
            print("Please fit the model first")
        pred_prob = self.predict_proba(X)
        pred = [1 if x > threshold else 0 for x in pred_prob]
        return pred

    def _cross_entropy(self, y, yhat):
        return -sum([y[i] * np.log2(yhat[i]) for i in range(len(y))])

    def evaluate(self, X, y):
        if not isinstance(self.weights, np.ndarray):
            print("Please fit the model first")
        yhat = self.predict_proba(X)
        results = []

        for i in range(len(y)):
            # create the distribution for each event {0, 1}
            expected = [1.0 - y[i], y[i]]
            predicted = [1.0 - yhat[i], yhat[i]]
            # calculate cross entropy for the two events
            ce = self._cross_entropy(expected, predicted)
            # print('>[y=%.1f, yhat=%.1f] ce: %.3f nats' % (y[i], yhat[i], ce))
            results.append(ce)

        yhat_array = np.array(self.predict(X))
        acc = np.mean(y == yhat_array)
        loss = self.loss_function(X, y)
        print(f"Accuracy = {acc}, Average Cross-Entopy = {np.mean(results)} nats")
        print(f"Loss = {loss}")
