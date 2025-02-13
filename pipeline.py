import torch
from ebm.util.config import Config
# from ebm.networks import get_network
from ebm.networks import DeepHopfieldNetwork
from ebm.estimators.gradient_estimators import EquiPropEstimator
from ebm.estimators.optimizer import SGDOptimizer
from ebm.estimators.cost import SquaredError
from ebm.runner import NetworkRunner

# Create a toy d


dataset = 'TwoMoons'
config = Config()

#network = get_network(config)
layers = [2,164,2] #config.layers
network = DeepHopfieldNetwork(layers,config)
cost_function = SquaredError(network)
optimizer = SGDOptimizer(network,cost_function)
estimator = EquiPropEstimator(network, cost_function)
# runner = NetworkRunner(network,estimator,optimizer, training_loader, test_loader)
# runner.train(num_epochs=100)
# runner.evaluate(custom=False)