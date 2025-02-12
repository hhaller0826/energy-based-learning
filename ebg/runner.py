import sys
import torch



class Epoch:
    """
    Class used to process a dataset with a network once (one 'epoch') to change the internal state of the network and/or compute statistics.
    Two important subclasses of Epoch are Evaluator and Trainer.

    Attributes
    ----------
    _stats (list of list of Statistic): the list of all lists of statistics to be computed over the dataset

    Methods
    -------
    add_statistic(stat, list_idx)
        Adds a statistic to the collection of index list_idx
    dataset_size()
        Returns the size of the dataset
    _reset()
        Sets all the statistics to zero
    _do_measurements(list_idx)
        Do the measurements for each of the statistics in list of index list_idx
    """

    def __init__(self, num_lists):
        """Creates an instance of Epoch

        Args:
            num_lists (int): the number of lists of statistics
        """

        self._stats = []
        for _ in range(num_lists): self._stats.append([])

    def add_statistic(self, stat, list_idx=0):
        """Adds a statistic to the list of statistics

        Args:
            stat (Statistic): the statistic to be added
            list_idx (int, optional): the index of the list in which we add the statistic. Default: 0
        """

        self._stats[list_idx].append(stat)

    def dataset_size(self):
        """Returns the size of the dataset"""
        return len(self._dataloader.dataset)

    def _all_stats(self):
        """Returns the list of all stats, both from the 'eval' and 'train' collections"""
        return [stat for list_stats in self._stats for stat in list_stats]

    def _reset(self):
        """Sets all the statistics to zero"""

        for stat in self._all_stats(): stat.reset()

    def _do_measurements(self, list_idx=0):
        """Do measurements in all stats of the collection of index list_idx

        Args:
            list_idx (int, optional): the index of the list where we measure all the statistic. Default: 0
        """

        for stat in self._stats[list_idx]: stat.do_measurement()

    def __str__(self):
        list_of_strings = [str(stat) for stat in self._all_stats() if stat.display]
        string = ', '.join(list_of_strings)
                
        return string



class Evaluator(Epoch):
    """
    Class used to evaluate a network on a dataset

    Attributes
    ----------
    _network (SumSeparableFunction): the model to evaluate
    _dataloader (Dataloader): the dataset on which to evaluate the model
    energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference

    idx (tensor of int): vector of indices of the data examples in the current mini-batch

    Methods
    -------
    run(verbose)
        Evaluates the network over the dataset
    """

    def __init__(self, network, cost_fn, dataloader, energy_minimizer):
        """Initializes an instance of Evaluator

        Args:
            network (Network): the model to evaluate
            cost_fn (CostFunction): the cost function to optimize
            dataloader (Dataloader): the dataset on which to evaluate the model. An IndexedDataset that loads data in the form of triplets (x, y, idx)
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference
        """

        Epoch.__init__(self, 1)

        self._network = network
        self._cost_fn = cost_fn
        self._dataloader = dataloader

        self._energy_minimizer = energy_minimizer

    @property
    def idx(self):
        """Gets the indices of the examples in the last mini batch processed"""
        return self._idx

    def run(self, verbose=False):
        """Evaluate the model over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after processing the entire dataset. Default: False.
        """

        self._reset()  # sets all the statistics to zero

        for x, y, idx in self._dataloader:

            # Inference (free phase relaxation)
            self._network.set_input(x, reset=True)
            self._energy_minimizer.compute_equilibrium()

            # Measure statistics
            self._idx = idx
            self._cost_fn.set_target(y)
            self._do_measurements()

            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(str(self))
            sys.stdout.write('\n')

    def __str__(self):
        return 'TEST  -- ' + Epoch.__str__(self)



class Trainer(Epoch):
    """
    Class used to train a network on a dataset

    Attributes
    ----------
    _network (SumSeparableFunction): the network to train
    _dataloader (Dataloader): the dataset on which to train the network
    _differentiator (GradientEstimator): the method used to train the network
    energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference

    _stats (list of Statistic): _stats[0] is the list of statistics to measure after inference (evaluation), and _stats[1] is the list of statistics to measure after computing the gradient (training)

    Methods
    -------
    run(verbose)
        Train the network for one epoch over the dataset
    """

    def __init__(self, network, cost_fn, params, dataloader, differentiator, optimizer, energy_minimizer):
        """Initializes an instance of Trainer

        Args:
            network (Network): the network to train
            cost_fn (CostFunction): the cost function to optimize
            dataloader (Dataloader): the dataset on which to train the network
            differentiator (GradientEstimator): either EquilibriumProp or Backprop
            optimizer (str): the optimizer used to optimize.
            energy_minimizer (EnergyMinimizer): the algorithm used to minimize the energy function at inference
        """


        Epoch.__init__(self, 2)

        self._network = network
        self._params = params + cost_fn.params()  # FIXME
        self._cost_fn = cost_fn
        self._dataloader = dataloader
        self._differentiator = differentiator
        self._optimizer = optimizer
        self._energy_minimizer = energy_minimizer

    def run(self, verbose=False):
        """Train the model for one epoch over the dataset.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: False.
        """

        self._reset()  # sets all the statistics to zero

        for x, y in self._dataloader:

            # inference (free phase relaxation)
            self._network.set_input(x, reset=False)  # we set the input, and we let the state of the network where it was at the end of the previous batch
            self._energy_minimizer.compute_equilibrium()  # we let the network settle to equilibrium (free state)
            self._cost_fn.set_target(y)  # we present the correct (desired) output
            self._do_measurements(0)  # we measure the statistics of the free state (energy value, cost value, error value, ...)

            # training step
            grads = self._differentiator.compute_gradient()  # compute the parameter gradients
            for param, grad in zip(self._params, grads): param.state.grad = grad  # Set the gradients of the parameters
            self._do_measurements(1)  # measure the statistics of training
            self._optimizer.step()  # perform one step of gradient descent on the parameters (of both the energy function E and the cost function C)
            for param in self._params: param.clamp_()  # clamp the parameters' states in their range of permissible values, if adequate

            if verbose:  # log the characteristics of training for the current epoch, up to the current mini-batch
                sys.stdout.write('\r')
                sys.stdout.write(str(self))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.write(str(self))
            sys.stdout.write('\n')

    def __str__(self):
        return 'TRAIN -- ' + Epoch.__str__(self)
    

from datetime import datetime
import pickle
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from training.statistics import Counter, ErrorFinder, EnergyStat, CostStat, ErrorStat, TopFiveErrorStat, ViolationStat, NormStat, SaturationStat, GradientStat


class Monitor:
    """
    Class used to monitor the training process.

    Attributes
    ----------
    _network (SumSeparableFunction): the network to train
    _trainer (Trainer): used to train the network on the training set
    _scheduler (lr_scheduler): used to adjust the learning rates after every training epoch
    _evaluator (Evaluator): used to evaluate the network on the test set
    _path (str): the directory where to save the model and the characteristics of the training process
    _series (list of TimeSeries): the time series of statistics that are monitored during training
    _test_error_curve (TimeSeries): the test error curve
    _epoch (int): epoch of training (where the statistics have been recorded)
    _writer (SummaryWriter): tensorboard summary writer to update

    Methods
    -------
    run(num_epochs, verbose=False, use_tensorboard=True)
        Trains the network for num_epochs epochs
    save_network()
        Saves the model in the path
    save_series()
        Saves the state of the training process in the path
    """

    def __init__(self, network, cost_fn, trainer, scheduler, evaluator, path=None, use_tensorboard=True):
        """Creates an instance of Monitor

        Args:
            network (SumSeparableFunction): the network to train
            cost_fn (CostFunction): the cost function to optimize
            trainer (Trainer): used to train the network on the training set
            evaluator (Evaluator): used to evaluate the network on the test set
            scheduler (lr_scheduler): used to adjust the learning rates after every training epoch
            path (str, optional): the directory where to save the model and the characteristics of the training process. Default: None
            use_tensorboard (bool, optional): if True, uses a summary writer to monitor with tensorboard. Default: True
        """

        self._network = network
        self._cost_fn = cost_fn
        self._trainer = trainer
        self._scheduler = scheduler
        self._evaluator = evaluator
        # self._gdd = gdd  # FIXME

        self._path = datetime.now().strftime("%Y%m%d-%H%M%S") if path is None else path  # If no path is given, uses the date of creation of the monitor

        self._series = []
        self._test_error_curve = None  # initialized in _build_series()

        self._build_series()

        self._epoch = 0

        self._use_tensorboard = use_tensorboard
        if use_tensorboard: self._writer = SummaryWriter(self._path)

        self._start_time = time.time()


    def run(self, num_epochs, verbose=False):
        """Launch a run for num_epochs epochs

        Logs statistics about the run either after every batch or after every epoch.

        Args:
            num_epochs (int): number of epochs of training
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: False
        """

        for _ in range(num_epochs): self.one_epoch(verbose=verbose)
    
    def one_epoch(self, verbose=False):
        """Performs one epoch of training

        Logs statistics about the run either after every batch or at the end of the epoch.

        Args:
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs at the end of the epoch. Default: False
        """

        self._epoch += 1

        print('Epoch {}'.format(self._epoch))

        # Training
        self._trainer.run(verbose)
        self._scheduler.step()

        # Evaluation
        self._evaluator.run(verbose)

        # Update the statistics of training and evaluation
        for series in self._series: series.update()
        if self._use_tensorboard: self._update_summary_writer()

        if not verbose:  # Print the statistics of training and evaluation at every epoch
            print(str(self._trainer))
            print(str(self._evaluator))
        
        self.save_series()  # saves the training curves
        if self._test_error_curve.is_minimum(): self.save_network()  # saves the network's parameters

        # Print the total duration
        seconds = time.time() - self._start_time
        minutes, seconds = seconds // 60, seconds % 60
        hours, minutes = minutes // 60, minutes % 60
        print('Duration = {:.0f} hours {:.0f} min {:.0f} sec \n'.format(hours, minutes, seconds))
    
    def test_error(self):
        """Returns the last value of the test error rate"""
        return self._test_error_curve.last_value()

    def save_network(self):
        """Saves the network's parameters"""

        model_path = self._path + '/model.pt'
        self._network.save(model_path)

    def save_series(self):
        """Saves the time series (`training curves')"""

        time_series_path = self._path + '/time_series.pkl'
        with open(time_series_path, 'wb') as handle:
            dictionary = {series.name: series.get() for series in self._series if series.name}
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _add_statistic(self, statistic, train, list_idx=0):
        """Adds a statistic to either the trainer or the evaluator, and adds the corresponding time series to the monitor

        Special cases:
        * if the statistic is a ChangeStat of a Layer, also adds the corresponding series to _series_layer_changes
        * if the statistic is a test ErrorStat, also sets the corresponding series as _test_error_curve

        Args:
            statistic (Statistic): the statistic to add
            train (bool): whether we add the statistic to the trainer (True) or the evaluator (False)
            list_idx (int, optional): the index of the list that we add the statistic to. Default: 0
        """

        if train: self._trainer.add_statistic(statistic, list_idx)
        else: self._evaluator.add_statistic(statistic)

        series = TimeSeries(statistic, train)
        self._series.append(series)

        if not train and isinstance(statistic, ErrorStat): self._test_error_curve = series

    def _build_series(self):
        """Prepares the statistics and the corresponding time series to monitor"""

        network = self._network
        cost_fn = self._cost_fn
        train_set_size = self._trainer.dataset_size()
        test_set_size = self._evaluator.dataset_size()

        # the statistics to add to the trainer
        stats_train_0 = [
        Counter(network, train_set_size),
        EnergyStat(network),
        CostStat(cost_fn),
        ErrorStat(cost_fn),
        TopFiveErrorStat(cost_fn),
        ]
        stats_train_0 += [NormStat(layer) for layer in network.layers()]
        stats_train_0 += [SaturationStat(layer) for layer in network.layers()]
        # stats_train_1 = [GradientStat(layer) for layer in network.layers()]  # FIXME
        stats_train_1 = [GradientStat(param) for param in network.params()]

        for stat in stats_train_0: self._add_statistic(stat, train=True)
        for stat in stats_train_1: self._add_statistic(stat, train=True, list_idx=1)

        # the statistics to add to the evaluator
        stats_test = [
        Counter(network, test_set_size),
        EnergyStat(network),
        CostStat(cost_fn),
        ErrorStat(cost_fn),
        TopFiveErrorStat(cost_fn),
        # ErrorFinder(network, evaluator),
        ]
        stats_test += [NormStat(layer) for layer in network.layers()]
        stats_test += [SaturationStat(layer) for layer in network.layers()]

        for stat in stats_test: self._add_statistic(stat, train=False)

    def _update_summary_writer(self):
        """Add the statistics to the summary writer to monitor with tensorboard"""

        # for param in self._network.params(): self._writer.add_histogram(param.name, param.state, self._epoch)
        # hanita

        for series in self._series:
            if series.name: self._writer.add_scalar(series.name, series.last_value(), self._epoch)

        # FIXME
        # figs = self._gdd.produce_curves()
        # for name, fig in figs.items(): self._writer.add_image('GDD {}'.format(name), fig, self._epoch)




class Runner:
    """
    Class used to run a training process.

    Attributes
    ----------
    _network (SumSeparableFunction): the network to train
    _cost_fn (CostFunction): the cost function to optimize
    _trainer (Trainer): used to train the network on the training set
    _scheduler (lr_scheduler): used to adjust the learning rates after every training epoch
    _evaluator (Evaluator): used to evaluate the network on the test set
    _monitor (Monitor): used to monitor the training process

    Methods
    -------
    run(num_epochs, verbose=False, use_tensorboard=True)
        Trains the network for num_epochs epochs
    """

    def __init__(self,train_dataloader,test_dataloader, model, estimator, scheduler , config='ebg/default_config.json'):
        """Creates an instance of Runner

        Args:
            network (SumSeparableFunction): the network to train
            cost_fn (CostFunction): the cost function to optimize
            trainer (Trainer): used to train the network on the training set
            evaluator (Evaluator): used to evaluate the network on the test set
            scheduler (lr_scheduler): used to adjust the learning rates after every training epoch
            path (str, optional): the directory where to save the model and the characteristics of the training process. Default: None
            use_tensorboard (bool, optional): if True, uses a summary writer to monitor with tensorboard. Default: True
        """

        self._network = network
        self._cost_fn = cost_fn
        self._trainer = trainer
        self._scheduler = scheduler
        self._evaluator = evaluator

        self._monitor = Monitor(network, cost_fn, trainer, scheduler, evaluator, path, use_tensorboard)

    def run(self, num_epochs, verbose=False):
        """Launch a run for num_epochs epochs

        Logs statistics about the run either after every batch or after every epoch.

        Args:
            num_epochs (int): number of epochs of training
            verbose (bool, optional): if True, prints logs after every batch processed ; if False: prints logs after every epoch. Default: False
        """

        self._monitor.run(num_epochs, verbose)