import torch
from config import Config
from data_loader import load_dataloaders
from model import get_model
from estimator import EquilibriumPropEstimator
from optimizer import Optimizer
from trainer import Trainer
from evaluator import Evaluator
from monitor import Monitor



dataset = 'TwoMoons'
training_loader, test_loader = load_dataloaders(dataset, config.batch_size, augment_32x32=False, normalize=False)
config = Config()
#network = get_network(config)
layers = [2,164,2] #config.layers
network = DeepHopfieldNetwork(layers,config)
cost_fn = SquaredError(network)
optimizer = Optimizer(network)
estimator = EquilibriumPropEstimator(network, cost_fn)
runner = NetworkRunner(network,estimator,optimizer, training_loader, test_loader)
runner.train(num_epochs=100)
runner.evaluate(custom=False)