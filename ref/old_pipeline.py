import argparse
import numpy
import torch

from datasets import load_dataloaders, load_mnist
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


# Layers and Layer shapes for the model
num_hidden_layers = 1
num_neurons = 64
model = 'dhn-'+str(num_hidden_layers)+'h_twomoons'
layer_shapes = []
layer_shapes.append((2,))
for i in range(num_hidden_layers): #TODO: implement layer adapter function 
    layer_shapes.append((num_neurons,))
layer_shapes.append((2,))

# Other params 
num_iterations_inference = 50
num_iterations_training = 20
nudging = 0.2
num_epochs = 25

# More params
weight_init_dist = 'xavier_uniform' # Need to implement function to easily allow replacement
weight_gains = [1.0] * (num_hidden_layers+1)
learning_rates_weights = list(np.linspace(0.2, 0.01, num_hidden_layers+1))
learning_rates_biases = list(np.linspace(0.2, 0.01, num_hidden_layers+1))

# Model and Energy function 


energy_fn = DeepHopfieldEnergy(layer_shapes, weight_gains, weight_init_dist)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
energy_fn.set_device(device) # Implement in class

output_layer = energy_fn.layers()[-1] # Output layer may not always be the last layer
cost_fn = SquaredError(output_layer)

network = Network(energy_fn) # Unify energy_fn and network same thing 

params = energy_fn.params() # Implement in class
layers = energy_fn.layers() # Implement in class
free_layers = network.free_layers() # Implement in class

augmented_fn = AugmentedFunction(energy_fn, cost_fn) # input only network apply check in case of custom 
energy_minimizer_training = FixedPointMinimizer(augmented_fn, free_layers) # input only network apply check in case of custom
estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training) # netowk, energy_fn, cost_fn, energy_minimizer_training as input only. Can I abstract it out further ???
estimator.nudging = nudging # Implement in class
estimator.variant = 'centered' # Implement in class

energy_minimizer_training.num_iterations = num_iterations_training
energy_minimizer_training.mode = 'asynchronous'

learning_rates = learning_rates_biases + learning_rates_weights
momentum = 0.
weight_decay = 0. * 1e-4
optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

energy_minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
energy_minimizer_inference.num_iterations = num_iterations_inference
energy_minimizer_inference.mode = 'asynchronous'

trainer = Trainer(network, cost_fn, params, training_loader, estimator, optimizer, energy_minimizer_inference)
evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
path = '/'.join(['papers/fast-drn', model, 'EP'])
monitor = Monitor(energy_fn, cost_fn, trainer, scheduler, evaluator, path)

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

tensors = []
tensor_names = []

for p in energy_fn.params():
    tensor_names.append(p.name)
    tensors.append(p.get())

fig, axes = plt.subplots(1, len(tensors), figsize=(15, 5))
for i, (tensor, ax) in enumerate(zip(tensors, axes)):
    if tensor.ndim > 1:
        tensor = tensor.flatten()
    ax.hist(tensor.numpy(), bins=100, alpha=0.7, color=f"C{i}")
    ax.set_title(tensor_names[i] + ' Initial Values')
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
fig.suptitle('Initial Parameter Values', fontsize=16)
# Adjust layout
plt.tight_layout()
plt.show()

eq_neuron_dict = {}

hidden_layers = network.free_layers()
hidden_layer_names = []
for layer in hidden_layers:
    hidden_layer_names.append(layer.name)
    eq_neuron_dict[layer.name] = []

print(hidden_layer_names)

eq_energies = []
all_data = []


for k, (x, y, idx) in enumerate(test_loader):
    batch = x
    all_data.append(batch)
    batch_labels = y
    network.set_input(batch, reset=True)
    en, n = energy_minimizer_inference.compute_eq_energy()
    eq_energies.append(en)
    for layer in hidden_layer_names:
        layer_neurons = n[layer]
        temp = eq_neuron_dict[layer] 
        temp.append(layer_neurons)
        eq_neuron_dict[layer] = temp

for layer in hidden_layer_names:
    neuron_list = eq_neuron_dict[layer]
    stacked_tensor = torch.cat(neuron_list, dim=0)
    eq_neuron_dict[layer] = stacked_tensor.numpy()

eq_energies = torch.cat(eq_energies, dim=0).numpy()
all_data = torch.cat(all_data, dim=0).numpy()
        

logit_out = eq_neuron_dict[hidden_layer_names[-1]]
exp_arr = np.exp(logit_out - np.max(logit_out, axis=1, keepdims=True)) 
softmax_probs = exp_arr / np.sum(exp_arr, axis=1, keepdims=True)


layer = 'Layer_1'

umap_model1 = umap.UMAP(n_components=2, random_state=42)
umap1 = umap_model1.fit_transform(eq_neuron_dict[layer])