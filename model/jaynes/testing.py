import numpy as np 

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
    # Create a dataset (X, target)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Example input
    target = np.array([[0], [1], [1], [0]])          # XOR problem

    # Initialize and train the network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    nn.equilibrium_propagation(X, target, iterations=1000)

    # Test the network
    print("Outputs after training:")
    for x in X:
        output = nn.forward(x.reshape(1, -1))
        print(f"Input: {x} => Output: {output}")