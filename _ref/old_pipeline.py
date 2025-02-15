
import argparse
import numpy
import torch
from model.hopfield.network import DeepHopfieldEnergy
from model.function.network import Network
from model.function.cost import SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.sgd import EquilibriumProp, Backprop, AugmentedFunction
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

import umap


# Create a toy dataset of 2 interlocking rings in 3D and with 1000 points and labels for each point
# Create a toy dataset of interlocking rings perpendicular to each other in 3D
num_points = 500
theta = torch.linspace(0, 2 * torch.pi, num_points)

# First ring in the XY plane, centered at (1, 0, 0)
x1 = 1 + torch.cos(theta)
y1 = torch.sin(theta)
z1 = torch.zeros(num_points)

# Second ring in the XZ plane, centered at (0, 0, 1)
x2 = torch.cos(theta)
y2 = torch.zeros(num_points)
z2 = 1 + torch.sin(theta)

# Combine the rings
X1 = torch.stack([x1, y1, z1], dim=1)
X2 = torch.stack([x2, y2, z2], dim=1)
X = torch.cat([X1, X2], dim=0)

# Labels for the rings
Y1 = torch.zeros(num_points, dtype=torch.int32)
Y2 = torch.ones(num_points, dtype=torch.int32)
Y = torch.cat([Y1, Y2], dim=0)


# %%
# Create a dataloader for the dataset
dataset = torch.utils.data.TensorDataset(X, Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# Split the dataset into a training and test set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# %% [markdown]
# # Output of the Network
# 
# There are two classes, 0 or 1, in our classification task. Ater performing inference on a specific input 2D point, the values of the network's output neurons will be interpreted as probabilities (After softmaxing) that the input point belongs to the respective classes. Therefore, our neural network will have two output neurons, one representing class 0, and the other representing class 1. 

# %%
num_hidden_layers = 1
num_neurons = 64

model = 'dhn-'+str(num_hidden_layers)+'h_twomoons'

layer_shapes = []
layer_shapes.append((3,))
for i in range(num_hidden_layers):
    layer_shapes.append((num_neurons,))
layer_shapes.append((2,))

# %%
num_iterations_inference = 50
num_iterations_training = 20
nudging = 0.2
num_epochs = 25

# %%
weight_init_dist = 'xavier_uniform'
weight_gains = [1.0] * (num_hidden_layers+1)
learning_rates_weights = list(np.linspace(0.2, 0.01, num_hidden_layers+1))
learning_rates_biases = list(np.linspace(0.2, 0.01, num_hidden_layers+1))

# %%


# %%
energy_fn = DeepHopfieldEnergy(layer_shapes, weight_gains, weight_init_dist)
if torch.cuda.is_available(): device = "cuda"
else: device = "cpu"
energy_fn.set_device(device)

output_layer = energy_fn.layers()[-1]
cost_fn = SquaredError(output_layer)

network = Network(energy_fn)

# %%
params = energy_fn.params()
layers = energy_fn.layers()
free_layers = network.free_layers()

# %%
Y = Y.long()
augmented_fn = AugmentedFunction(energy_fn, cost_fn)
energy_minimizer_training = FixedPointMinimizer(augmented_fn, free_layers)
estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
estimator.nudging = nudging
estimator.variant = 'centered'

energy_minimizer_training.num_iterations = num_iterations_training
energy_minimizer_training.mode = 'asynchronous'

learning_rates = learning_rates_biases + learning_rates_weights
momentum = 0.
weight_decay = 0. * 1e-4
optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

energy_minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
energy_minimizer_inference.num_iterations = num_iterations_inference
energy_minimizer_inference.mode = 'asynchronous'

trainer = Trainer(network, cost_fn, params, train_loader, estimator, optimizer, energy_minimizer_inference)
evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
path = '/'.join(['papers/fast-drn', model, 'EP'])
monitor = Monitor(energy_fn, cost_fn, trainer, scheduler, evaluator, path)
batch_size = 32
print('Dataset: {} -- batch_size={}'.format(dataset, batch_size))
print('Network: ', energy_fn)
print('Cost function: ', cost_fn)
print('Energy minimizer during inference: ', energy_minimizer_inference)
print('Energy minimizer during training: ', energy_minimizer_training)
print('Gradient estimator: ', estimator)
print('Parameter optimizer: ', optimizer)
print('Number of epochs = {}'.format(num_epochs))
print('Path = {}'.format(path))
print('Device = {}'.format(device))
print()

# %%
monitor.run(num_epochs, verbose=True)