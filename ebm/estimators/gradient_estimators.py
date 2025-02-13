# update config function import
import torch
from ebm.estimators.optimizer import Optimizer
from ebm.estimators.cost import SquaredError
from ebm.estimators.augmented_function import AugmentedFunction, FixedPointMinimizer
from ebm.estimators.equilibrium_prop import EquilibriumProp


def _get_cost_fnc(model, config):
    if config.cost_function['name'] == 'squared_error':
        return SquaredError(model, config)

def _get_augmented_fnc(model, config,cost_fnc):
    if config.augmented_function['name'] == 'fixed_point':
        return AugmentedFunction(model, cost_fnc)

def _get_energy_minimizer(config, augmented_fnc, free_layers):
    if config.minimizer['name'] == 'fixed_point':
        return FixedPointMinimizer(augmented_fnc,free_layers)

class EquiPropEstimator:
    def __init__(self, model, cost_fnc, config, augmented_fnc=None, energy_minimizer=None):
        self.energy_fn = model.function()
        if cost_fnc is None:
            self.cost_fnc = _get_cost_fnc(model, config)
        else:
            self.cost_fnc = cost_fnc
        if augmented_fnc is None:    
            self.augmented_fnc = _get_augmented_fnc(model, config, cost_fnc)
        else:
            self.augmented_fnc = augmented_fnc
        if energy_minimizer is None:
            self.energy_minimizer = _get_energy_minimizer(config, augmented_fnc, model.free_layers())
        else:
            self.energy_minimizer = energy_minimizer
        self.free_layers = model.free_layers() 
        self.params = energy_fn.params() 
        self.layers = energy_fn.layers() 
        self.gradient_estimator = EquilibriumProp(self.params, self.layers, self.augmented_fnc, self.cost_fnc, self.energy_minimizer) 


