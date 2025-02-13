import sys
import time
import pickle
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

# These imports are assumed to exist in your code base.
from util.statistics import (Counter, ErrorFinder, EnergyStat, CostStat,
                                 ErrorStat, TopFiveErrorStat, ViolationStat,
                                 NormStat, SaturationStat, GradientStat)
# Also assume you have a TimeSeries class for logging statistics.
from util.timeseries import TimeSeries  # <-- adjust as needed


class NetworkRunner:
    """
    A unified class that handles training, evaluation, and progress monitoring.
    
    Public methods:
      - train(num_epochs, verbose=False): Runs training (and evaluation at the end of each epoch)
      - eval(dataloader=None, verbose=False): Runs evaluation on the default evaluation dataloader
          (or on a provided one)
    """
    def __init__(self, model,optimizer,estimator, train_dataloader, eval_dataloader,config
                 params=None, differentiator=None, optimizer=None, energy_minimizer=None,
                 scheduler=None, path=None, use_tensorboard=True):
        """
        Initialize the ModelRunner with all the ingredients.
        
        Args:
            network: the model (assumed to have methods like set_input, save, layers, params, etc.)
            cost_fn: the cost function (with method set_target and params())
            train_dataloader: dataloader for training (yields (x, y) examples)
            eval_dataloader: dataloader for evaluation (yields either (x, y) or (x, y, idx))
            params: list of the network’s parameters (or objects that have a .state)
            differentiator: object that implements compute_gradient()
            optimizer: optimizer instance (has a .step() method)
            energy_minimizer: object that implements compute_equilibrium()
            scheduler: learning-rate scheduler (with a .step() method)
            path (str, optional): directory to save the model and training curves.
              If not provided, a folder name based on the current date/time is used.
            use_tensorboard (bool, optional): if True, logs statistics using a SummaryWriter.
        """
        self._network = model
        self._config = model.config
        self._estimator = estimator
        self._train_loader = train_dataloader
        self._eval_loader = eval_dataloader

        # Combine provided parameters with those returned by cost_fn.params()
        self._params = params + cost_fn.params()

        self._differentiator = estimator.gradient_estimator
        self._cost_fn = estimator.cost_fn
        self._optimizer = est
        self._energy_minimizer = energy_minimizer
        self._scheduler = scheduler

        self._path = datetime.now().strftime("%Y%m%d-%H%M%S") if path is None else path
        self._use_tensorboard = use_tensorboard
        if use_tensorboard:
            self._writer = SummaryWriter(self._path)

        self._epoch = 0
        self._start_time = time.time()

        # We build two sets of statistics: one for training and one for evaluation.
        # (The statistics are organized in “lists”: list 0 for free-phase measurements,
        #  list 1 for gradient-related measurements.)
        self._train_stats = {0: [], 1: []}
        self._eval_stats = []
        self._series = []        # TimeSeries objects for logging
        self._test_error_curve = None  # Will be set to the evaluation ErrorStat's series

        self._build_series()

    def _build_series(self):
        """
        Prepares all the statistics (and their corresponding time series)
        that will be updated during training and evaluation.
        """
        network = self._network
        cost_fn = self._cost_fn
        train_set_size = len(self._train_loader.dataset)
        test_set_size = len(self._eval_loader.dataset)

        # Statistics to measure after the free-phase (in training)
        stats_train_free = [
            Counter(network, train_set_size),
            EnergyStat(network),
            CostStat(cost_fn),
            ErrorStat(cost_fn),
            TopFiveErrorStat(cost_fn),
        ]
        # Add statistics for each layer (e.g., norm and saturation)
        stats_train_free += [NormStat(layer) for layer in network.layers()]
        stats_train_free += [SaturationStat(layer) for layer in network.layers()]

        # Statistics to measure after computing the gradients (in training)
        stats_train_grad = [GradientStat(param) for param in network.params()]

        for stat in stats_train_free:
            self._add_statistic(stat, train=True, list_idx=0)
        for stat in stats_train_grad:
            self._add_statistic(stat, train=True, list_idx=1)

        # Statistics for evaluation
        stats_eval = [
            Counter(network, test_set_size),
            EnergyStat(network),
            CostStat(cost_fn),
            ErrorStat(cost_fn),
            TopFiveErrorStat(cost_fn),
        ]
        stats_eval += [NormStat(layer) for layer in network.layers()]
        stats_eval += [SaturationStat(layer) for layer in network.layers()]

        for stat in stats_eval:
            self._add_statistic(stat, train=False)

    def _add_statistic(self, statistic, train, list_idx=0):
        """
        Adds a statistic to the appropriate collection (training or evaluation)
        and creates a corresponding TimeSeries instance.
        """
        if train:
            self._train_stats[list_idx].append(statistic)
        else:
            self._eval_stats.append(statistic)

        series = TimeSeries(statistic, train)
        self._series.append(series)

        # If this is an evaluation error statistic, store its time series (for model saving decisions, etc.)
        if (not train) and isinstance(statistic, ErrorStat):
            self._test_error_curve = series

    def _reset_statistics(self, train=True, list_idx=0):
        """
        Resets the statistics to zero for the given collection.
        """
        if train:
            for stat in self._train_stats[list_idx]:
                stat.reset()
        else:
            for stat in self._eval_stats:
                stat.reset()

    def _do_measurements(self, train=True, list_idx=0):
        """
        Performs the “do_measurement” call for each statistic in the given collection.
        """
        if train:
            for stat in self._train_stats[list_idx]:
                stat.do_measurement()
        else:
            for stat in self._eval_stats:
                stat.do_measurement()

    def _stats_str(self, train=True):
        """
        Returns a string summarizing the current statistics.
        """
        if train:
            all_stats = []
            for stat_list in self._train_stats.values():
                for stat in stat_list:
                    if getattr(stat, 'display', False):
                        all_stats.append(str(stat))
            return ', '.join(all_stats)
        else:
            return ', '.join(str(stat) for stat in self._eval_stats if getattr(stat, 'display', False))

    def _update_summary_writer(self):
        """
        Logs all time series data to tensorboard.
        """
        for series in self._series:
            if series.name:
                self._writer.add_scalar(series.name, series.last_value(), self._epoch)

    def save_network(self):
        """
        Saves the current network (model) parameters.
        """
        model_path = self._path + '/model.pt'
        self._network.save(model_path)

    def save_series(self):
        """
        Saves the training curves (time series of statistics) as a pickle file.
        """
        time_series_path = self._path + '/time_series.pkl'
        with open(time_series_path, 'wb') as handle:
            dictionary = {series.name: series.get() for series in self._series if series.name}
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train_epoch(self, verbose=False):
        """
        Runs one full epoch of training.
        (A single epoch loops over the training dataloader.)
        """
        # Reset training statistics (both free-phase and gradient-phase)
        self._reset_statistics(train=True, list_idx=0)
        self._reset_statistics(train=True, list_idx=1)

        for x, y in self._train_loader:
            # --- Inference (free phase relaxation) ---
            # (Note: here we use reset=False so that the network’s state “carries over”
            # from one mini-batch to the next if desired.)
            self._network.set_input(x, reset=False)
            self._energy_minimizer.compute_equilibrium()
            self._cost_fn.set_target(y)
            self._do_measurements(train=True, list_idx=0)
            
            # --- Training step (gradient computation and parameter update) ---
            grads = self._differentiator.compute_gradient()
            for param, grad in zip(self._params, grads):
                param.state.grad = grad  # assign the computed gradient to each parameter
            self._do_measurements(train=True, list_idx=1)
            self._optimizer.step()
            for param in self._params:
                param.clamp_()  # ensure the parameter values stay within allowed bounds
            
            if verbose:
                sys.stdout.write('\rTRAIN -- ' + self._stats_str(train=True))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\n')

    def eval_epoch(self, dataloader=None, verbose=False):
        """
        Runs evaluation over the entire evaluation set (or a provided dataloader).
        """
        if dataloader is None:
            dataloader = self._eval_loader

        self._reset_statistics(train=False)
        for batch in dataloader:
            # Depending on your dataloader, a batch may have three entries (x, y, idx)
            try:
                x, y, idx = batch
            except ValueError:
                x, y = batch
                idx = None

            self._network.set_input(x, reset=True)
            self._energy_minimizer.compute_equilibrium()
            self._cost_fn.set_target(y)
            self._do_measurements(train=False)
            if verbose:
                sys.stdout.write('\rEVAL -- ' + self._stats_str(train=False))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\n')

    def train(self, num_epochs, verbose=False):
        """
        Runs training for a given number of epochs. At the end of each epoch the scheduler
        is updated, evaluation is run, statistics are updated and (if appropriate) the network is saved.
        
        Args:
            num_epochs (int): number of epochs to train.
            verbose (bool, optional): if True, print progress details at every batch.
        """
        for _ in range(num_epochs):
            self._epoch += 1
            print('Epoch {}'.format(self._epoch))
            
            # Run one training epoch
            self.train_epoch(verbose=verbose)
            
            # Update the learning rate
            self._scheduler.step()
            
            # Run evaluation (using the default eval dataloader)
            self.eval_epoch(verbose=verbose)
            
            # Update all time series statistics
            for series in self._series:
                series.update()
            if self._use_tensorboard:
                self._update_summary_writer()
            
            # Optionally print the statistics if not already in verbose mode
            if not verbose:
                print('TRAIN -- ' + self._stats_str(train=True))
                print('EVAL  -- ' + self._stats_str(train=False))
            
            # Save the time series curves
            self.save_series()
            # Optionally save the network if the test error (as measured by ErrorStat) is minimal
            if self._test_error_curve and self._test_error_curve.is_minimum():
                self.save_network()
            
            # Print elapsed time
            elapsed = time.time() - self._start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            print('Duration = {:.0f} hours {:.0f} min {:.0f} sec \n'.format(hours, minutes, seconds))

    def eval(self, dataloader=None, verbose=False):
        """
        Runs evaluation over the default evaluation dataloader or over a custom one if provided.
        
        Args:
            dataloader: if provided, evaluation is run on this dataloader instead.
            verbose (bool, optional): if True, prints progress details.
        """
        self.eval_epoch(dataloader=dataloader, verbose=verbose)

    def __str__(self):
        return ('Epoch {}:\nTRAIN -- {}\nEVAL  -- {}'
                .format(self._epoch, self._stats_str(train=True), self._stats_str(train=False)))

