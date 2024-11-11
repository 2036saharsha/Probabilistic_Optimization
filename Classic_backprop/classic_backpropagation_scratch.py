import numpy as np
from sklearn.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def softmax(self, z):
        # Softmax activation function
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        # Cross-Entropy loss function
        n_samples = y_true.shape[0]
        logp = -np.log(y_pred[range(n_samples), y_true.argmax(axis=1)])
        return np.sum(logp) / n_samples

    def forward(self, X):
        # Forward pass
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)

    def backward(self, X, y_true, y_pred):
        # Backward pass (calculate gradients)
        n_samples = X.shape[0]
        dz = (y_pred - y_true) / n_samples  # Derivative of loss w.r.t z
        dW = np.dot(X.T, dz) 
        db = np.sum(dz, axis=0, keepdims=True)
        return dW, db

    def update_weights(self, dW, db):
        # Update weights and biases
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def train(self, X, y, epochs=500):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute the loss
            loss = self.cross_entropy_loss(y, y_pred)

            # Backward pass
            dW, db = self.backward(X, y, y_pred)

            # Update weights
            self.update_weights(dW, db)

            # Print the loss and accuracy every 50 epochs
            if epoch % 50 == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def evaluate(self, X, y):
        # Make predictions and calculate accuracy
        y_pred = self.forward(X)
        y_pred_labels = y_pred.argmax(axis=1)
        y_true_labels = y.argmax(axis=1)
        return accuracy_score(y_true_labels, y_pred_labels)

