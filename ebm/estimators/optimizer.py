import torch
def get_optimizer(model, cost_function):
    """Returns an optimizer for the model and cost function.

    Args:
        model (Model): the model whose parameters we optimize
        cost_function (CostFunction): the cost function whose parameters we optimize

    Returns:
        Optimizer: the optimizer
    """
    config = model.config
    if config.optimizer == 'sgd':
        return SGDOptimizer(model, cost_function)
    elif config.optimizer == 'adam':
        return AdamOptimizer(model, cost_function)
    else:
        raise ValueError('Unknown optimizer: {}'.format(config.optimizer))


class SGDOptimizer(torch.optim.SGD):

    def __init__(self, model, cost_function):
        """Creates an instance of the SGD Optimizer.

        Args:
            energy_fn (SumSeparableFunction): the energy function whose parameters we optimize
            cost_function (CostFunction): the cost function whose parameters we optimize
            learning_rates (list of float): the list of learning rates for the energy parameters and cost parameters
            momentum (float, optional): the momentum. Default: 0.0
            weight_decay (float, optional): the weight decay. Default: 0.0
        """
        energy_fn = model.function
        config = model.config
        learning_rates = config.optimizer['learning_rates_weights'] + config.optimizer['learning_rates_biases']
        momentum = config.optimizer["momentum"]
        weight_decay = config.optimizer["weight_decay"]

        self._learning_rates = learning_rates
        self._momentum = momentum
        self._weight_decay = weight_decay

        params = energy_fn.params() + cost_function.params()
        params = [{"params": param.state, "lr": lr} for param, lr in zip(params, learning_rates)]
        torch.optim.SGD.__init__(self, params, lr=0.1, momentum=momentum, weight_decay=weight_decay)

    def __str__(self):
        return 'SGD -- initial learning rates = {}, momentum={}, weight_decay={}'.format(self._learning_rates, self._momentum, self._weight_decay)

class AdamOptimizer(torch.optim.Adam):

    def __init__(self, model, cost_function):
        """Creates an instance of the Adam Optimizer.

        Args:
            energy_fn (SumSeparableFunction): the energy function whose parameters we optimize
            cost_function (CostFunction): the cost function whose parameters we optimize
            learning_rates (list of float): the list of learning rates for the energy parameters and cost parameters
            momentum (float, optional): the momentum. Default: 0.0
            weight_decay (float, optional): the weight decay. Default: 0.0
        """
        energy_fn = model._function()
        config = model.config
        learning_rates = config.learning_rates
        momentum = config.momentum
        weight_decay = config.weight_decay

        self._learning_rates = learning_rates
        self._momentum = momentum
        self._weight_decay = weight_decay

        params = energy_fn.params() + cost_function.params()
        params = [{"params": param.state, "lr": lr} for param, lr in zip(params, learning_rates)]
        torch.optim.Adam.__init__(self, params, lr=0.1, momentum=momentum, weight_decay=weight_decay)

    def __str__(self):
        return 'Adam -- initial learning rates = {}, momentum={}, weight_decay={}'.format(self._learning_rates, self._momentum, self._weight_decay)