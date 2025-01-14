import numpy as np 
import torch
from layer import *
from energy import *
from entropy import *

# I asked ChatGPT to code eq prop lol 
# might be good for basic testing
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01

    def forward(self, X):
        # Forward pass
        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.output = self.sigmoid(self.z2)
        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def energy(self, output, target):
        # Simple energy function (mean squared error)
        return np.mean((output - target) ** 2)

    def equilibrium_propagation(self, X, target, iterations=100):
        # Simulate equilibrium dynamics
        for _ in range(iterations):
            output = self.forward(X)

            # Compute energy
            energy = self.energy(output, target)

            # Here we would normally apply some dynamic process to update weights.
            # For simplicity, we'll just compute gradients and update weights directly.

            # Gradient estimation based on energy change
            dE_dz2 = 2 * (output - target) / output.shape[0]  # Gradient of energy w.r.t output
            dE_dW2 = np.dot(self.a1.T, dE_dz2 * self.sigmoid_derivative(self.z2))

            dE_da1 = np.dot(dE_dz2, self.W2.T)
            dE_dz1 = dE_da1 * self.sigmoid_derivative(self.z1)
            dE_dW1 = np.dot(X.T, dE_dz1)

            # Update weights using gradient descent
            learning_rate = 0.01
            self.W1 -= learning_rate * dE_dW1
            self.W2 -= learning_rate * dE_dW2

    def sigmoid_derivative(self, x):
        sig = self.sigmoid(x)
        return sig * (1 - sig)

# Example usage
if __name__ == "__main__":
    # # Create a dataset (X, target)
    # X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Example input
    # target = np.array([[0], [1], [1], [0]])          # XOR problem

    # # Initialize and train the network
    # nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    # nn.equilibrium_propagation(X, target, iterations=1000)

    # # Test the network
    # print("Outputs after training:")
    # for x in X:
    #     output = nn.forward(x.reshape(1, -1))
    #     print(f"Input: {x} => Output: {output}")

    sample_weights = np.random.uniform(low=0.0, high=1.0, size=(100,))
    print(sample_weights.sort())

    class DummyParam(Parameter):
        def __init__(self, tensor):
            Parameter.__init__(self, (1,1), None)
            self._state = tensor

        def init_state(self):
            pass


    def compare_methods(weights, num_bins:int=10, alpha:float=1., beta:float=1.):
        tensor = torch.Tensor(weights)
        
        params = DummyParam(tensor)

        bin_entropy = BinEntropy(tensor, num_bins)
        softargmax_entropy = SoftArgmaxEntropy(tensor)
        connection_layer = JaynesConnectionLayer(tensor, num_bins)


        print("=== {} BINS ===".format(num_bins))

        print("OG Potential: ", round(connection_layer.potential(alpha, beta),4))
        print("New Potential: ", round(jaynes_layer_potential(params, num_bins, alpha, beta),4))
        
        print("Entropy:", round(connection_layer.entropy(),4))
        print("BIN Entropy:", round(bin_entropy.eval(),4))
        print("SOFTARGMAX Entropy:", round(softargmax_entropy.eval(),4))
        print()

    

    compare_methods(sample_weights)
    compare_methods(sample_weights, 50)
    compare_methods(sample_weights, 100)