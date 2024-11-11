import numpy as np
from sklearn.metrics import accuracy_score

class PBPNeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        # Initialize mean and variance for each weight and bias
        self.W_mean = np.random.randn(input_size, output_size)
        self.W_var = np.full((input_size, output_size), 1.0)  # Start with high variance
        self.b_mean = np.zeros((1, output_size))
        self.b_var = np.full((1, output_size), 1.0)
        self.learning_rate = learning_rate

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        # Forward pass with Gaussian propagation of means and variances
        z_mean = X @ self.W_mean + self.b_mean
        z_var = X @ self.W_var + self.b_var  # Approximate variance propagation
        return self.softmax(z_mean), z_mean, z_var

    def cross_entropy_loss(self, y_true, y_pred):
        n_samples = y_true.shape[0]
        logp = -np.log(y_pred[range(n_samples), y_true.argmax(axis=1)])
        return np.sum(logp) / n_samples

    def backward(self, X, y_true, y_pred):
        # Derivative of the loss with respect to z (mean)
        n_samples = X.shape[0]
        dz = (y_pred - y_true) / n_samples  # Cross-entropy gradient
        dW_mean = X.T @ dz
        db_mean = np.sum(dz, axis=0, keepdims=True)

        # Update mean and variance of weights based on probabilistic gradient
        W_new_mean = self.W_mean - self.learning_rate * dW_mean
        W_new_var = self.W_var - self.learning_rate * (self.W_var * (dW_mean ** 2))

        b_new_mean = self.b_mean - self.learning_rate * db_mean
        b_new_var = self.b_var - self.learning_rate * (self.b_var * (db_mean ** 2))

        # Ensure variances stay positive and are clipped
        W_new_var = np.clip(W_new_var, 1e-6, None)
        b_new_var = np.clip(b_new_var, 1e-6, None)
        
        return W_new_mean, W_new_var, b_new_mean, b_new_var

    def update_weights(self, W_new_mean, W_new_var, b_new_mean, b_new_var):
        # Apply updates to means and variances
        self.W_mean, self.W_var = W_new_mean, W_new_var
        self.b_mean, self.b_var = b_new_mean, b_new_var

    def train(self, X, y, epochs=500):
        for epoch in range(epochs):
            # Forward pass
            y_pred, z_mean, z_var = self.forward(X)

            # Compute the loss
            loss = self.cross_entropy_loss(y, y_pred)

            # Backward pass with stabilized probabilistic gradients
            W_new_mean, W_new_var, b_new_mean, b_new_var = self.backward(X, y, y_pred)

            # Update weights and variances
            self.update_weights(W_new_mean, W_new_var, b_new_mean, b_new_var)

            # Print loss and accuracy every 50 epochs
            if epoch % 50 == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def evaluate(self, X, y):
        y_pred, _, _ = self.forward(X)
        y_pred_labels = y_pred.argmax(axis=1)
        y_true_labels = y.argmax(axis=1)
        return accuracy_score(y_true_labels, y_pred_labels)
