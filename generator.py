import utils
import numpy as np

class Generator:
  def __init__(self, input_dim, output_dim):
    self.weights = utils.initialize_weights((input_dim, output_dim))
    self.bias = np.zeros((1, output_dim))

  def forward(self, z):
    self.z = z
    self.hidden = np.dot(z, self.weights) + self.bias
    self.output = np.dot(z, self.weights) + self.bias
    return self.output
  
  def backward(self, grad_output, lr):
    grad_weights = np.dot(self.z.T, grad_output)  # Gradients for weights
    grad_bias = np.sum(grad_output, axis=0, keepdims=True)  # Gradients for biases

    # Update weights and biases
    self.weights -= lr * grad_weights
    self.bias -= lr * grad_bias
    #! Regularização: caso os weights sejam muito grandes, descomentar isto para limitar o valor deles
    #self.weights = np.clip(self.weights, -0.01, 0.01)