import utils
import numpy as np

class Discriminator:
    def __init__(self, input_dim):
        self.weights = utils.initialize_weights((input_dim, 1))
        self.bias = np.zeros((1, 1))

    def forward(self, x):
        self.x = x
        self.logits = np.dot(x, self.weights) + self.bias
        self.output = utils.sigmoid(self.logits)
        return self.output
    
    def backward(self, grad_output, lr):
        grad_weights = np.dot(self.x.T, grad_output)  # Gradients for weights
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # Update weights and biases
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias