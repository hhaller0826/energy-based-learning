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

dataset = 'TwoMoons'
training_loader, test_loader = load_dataloaders(dataset, config.batch_size, augment_32x32=False, normalize=False)

config = Config()
model = get_model(config)
# cost_fn = SquaredError(model,config)
# augmented_fn = AugmentedFunction(model, cost_fn) # input only network apply check in case of custom 
# energy_minimizer_training = FixedPointMinimizer(model, augmented_fn) # input only network apply check in case of custom energy_minimizer_inference = FixedPointMinimizer(model, config)
estimator = EquilibriumProp(model, config) # netowk, energy_fn, cost_fn, energy_minimizer_training as input only. Can I abstract it out further ???

optimizer = Optimizer(model, config) # cost_fn, 



scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

trainer = Trainer(training_loader,model, estimator, optimizer)
evaluator = Evaluator(model, estimator.cost_fn, test_loader)
monitor = Monitor(model, estimator.cost_fn, trainer, scheduler, evaluator, config)
monitor.run(config.num_epochs, verbose=True)

monitor.plot()

