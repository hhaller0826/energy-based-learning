import os
import json
import logging
import argparse
import sys
import time
import pickle
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import h5py  # Added for HDF5 support

# These imports are assumed to exist in your code base.
from util.statistics import (Counter, ErrorFinder, EnergyStat, CostStat,
                             ErrorStat, TopFiveErrorStat, ViolationStat,
                             NormStat, SaturationStat, GradientStat)
from util.timeseries import TimeSeries  # <-- adjust as needed

# -----------------------------------------------------------------------------
# Functions from the first file (configuration, logging, and results setup)
# -----------------------------------------------------------------------------

def create_dir(path):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def results(config):
    """
    Sets up directories and files for storing training results.
    """
    # Construct a unique result path using parameters from config.
    result_path = os.path.join(
        config['result_path'],
        f"{config['detection_model']}_{config['main_model']}_{config['dataset']}_{config['tag']}"
    )
    create_dir(result_path)

    # Initialize file logging.
    logging.basicConfig(
        filename=os.path.join(result_path, 'results.log'),
        level=logging.INFO
    )

    # Create a directory to save model weights.
    model_path = os.path.join(result_path, 'model_weights')
    create_dir(model_path)

    # Save the configuration to a JSON file.
    with open(os.path.join(result_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Update config with paths.
    config['best_model_path'] = os.path.join(result_path, 'best_model.pth')
    config['model_path'] = model_path

    return config

def set_config(args):
    """
    Loads config from a JSON file and overrides with command-line arguments.
    Also initializes wandb and sets up result directories.
    """
    with open(args.config) as f:
        config = json.load(f)

    # Override with CLI arguments if provided.
    for arg in vars(args):
        value = getattr(args, arg)
        if value is not None:
            config[arg] = value

    # If debug mode, disable wandb and file logging.
    if config.get('debug', False):
        config.update({'wandb': False, 'log': False})

    # Initialize wandb if enabled.
    if config.get('wandb', False):
        wandb.init(project='bad_content_detection', config=config)
        if config.get('tag', ''):
            wandb.run.tags = [config['tag']]

    # Set up directories and logging.
    config = results(config)
    return config

# -----------------------------------------------------------------------------
# The complete NetworkRunner class from the second file (with added HDF5 storage)
# -----------------------------------------------------------------------------

class NetworkRunner:
    """
    A unified class that handles training, evaluation, and progress monitoring.
    
    Public methods:
      - train(num_epochs, verbose=False): Runs training (and evaluation at the end of each epoch)
      - eval(dataloader=None, verbose=False): Runs evaluation on the default evaluation dataloader
          (or on a provided one)
      - store_weights_and_energy(test_loader): Reads and stores network weights and equilibrium energy
          in a structured .hdf5 file in the results directory.
    """
    def __init__(self, model, optimizer, estimator, scheduler, train_dataloader, eval_dataloader, config,
                 params=None, differentiator=None, energy_minimizer=None, path=None, use_tensorboard=True):
        """
        Initialize the NetworkRunner with all required ingredients.
        """
        # Instead of using model.config, we now pass in the unified config.
        self._network = model
        self._config = config  
        self._estimator = estimator
        self._train_loader = train_dataloader
        self._eval_loader = eval_dataloader
        self._cost_function = estimator.cost_function

        # Combine provided parameters with those returned by cost_fn.params()
        self._params = self._network.params + self._cost_function.params()

        self._differentiator = estimator.gradient_estimator
        self._cost_fn = estimator.cost_fn
        self._optimizer = optimizer
        self._energy_minimizer = estimator.energy_minimizer
        self._scheduler = scheduler
        
        # Use config-provided path if available; otherwise generate one from the current datetime.
        path = self._config.get('path', None)
        self._path = path if path is not None else datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self._use_tensorboard = self._config.get('use_tensorboard', True)
        if self._config.get('training', {}).get('use_tensorboard', False):
            self._writer = SummaryWriter(self._path)

        self._epoch = 0
        self._start_time = time.time()

        # Build statistics collections.
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

        # Statistics for the free-phase during training.
        stats_train_free = [
            Counter(network, train_set_size),
            EnergyStat(network),
            CostStat(cost_fn),
            ErrorStat(cost_fn),
            TopFiveErrorStat(cost_fn),
        ]
        # Add per-layer statistics.
        stats_train_free += [NormStat(layer) for layer in network.layers()]
        stats_train_free += [SaturationStat(layer) for layer in network.layers()]

        # Statistics for the gradient-phase during training.
        stats_train_grad = [GradientStat(param) for param in network.params()]

        for stat in stats_train_free:
            self._add_statistic(stat, train=True, list_idx=0)
        for stat in stats_train_grad:
            self._add_statistic(stat, train=True, list_idx=1)

        # Evaluation statistics.
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

        # Store the evaluation error series for potential model saving.
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
        Performs the measurement call for each statistic in the given collection.
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
        Logs all time series data to TensorBoard.
        """
        for series in self._series:
            if series.name:
                self._writer.add_scalar(series.name, series.last_value(), self._epoch)

    def save_network(self):
        """
        Saves the current network (model) parameters.
        """
        model_path = os.path.join(self._path, 'model.pt')
        self._network.save(model_path)

    def save_series(self):
        """
        Saves the training curves (time series of statistics) as a pickle file.
        """
        time_series_path = os.path.join(self._path, 'time_series.pkl')
        with open(time_series_path, 'wb') as handle:
            dictionary = {series.name: series.get() for series in self._series if series.name}
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train_epoch(self, verbose=False):
        """
        Runs one full epoch of training.
        (A single epoch loops over the training dataloader.)
        """
        # Reset statistics for both free-phase and gradient-phase.
        self._reset_statistics(train=True, list_idx=0)
        self._reset_statistics(train=True, list_idx=1)

        for x, y in self._train_loader:
            # --- Inference (free phase relaxation) ---
            self._network.set_input(x, reset=False)
            self._energy_minimizer.compute_equilibrium()
            self._cost_fn.set_target(y)
            self._do_measurements(train=True, list_idx=0)
            
            # --- Training step (gradient computation and parameter update) ---
            grads = self._differentiator.compute_gradient()
            for param, grad in zip(self._params, grads):
                param.state.grad = grad  # assign computed gradients
            self._do_measurements(train=True, list_idx=1)
            self._optimizer.step()
            for param in self._params:
                param.clamp_()  # ensure parameters remain within bounds
            
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
            # A batch may have three entries (x, y, idx) or two.
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
        """
        for _ in range(num_epochs):
            self._epoch += 1
            print('Epoch {}'.format(self._epoch))
            
            # Run one training epoch.
            self.train_epoch(verbose=verbose)
            
            # Update learning rate.
            self._scheduler.step()
            
            # Run evaluation.
            self.eval_epoch(verbose=verbose)
            
            # Update all time series statistics.
            for series in self._series:
                series.update()
            if self._use_tensorboard:
                self._update_summary_writer()
            
            if not verbose:
                print('TRAIN -- ' + self._stats_str(train=True))
                print('EVAL  -- ' + self._stats_str(train=False))
            
            # Save the training curves.
            self.save_series()
            # Save the network if the evaluation error is at a minimum.
            if self._test_error_curve and self._test_error_curve.is_minimum():
                self.save_network()
            
            elapsed = time.time() - self._start_time
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            print('Duration = {:.0f} hours {:.0f} min {:.0f} sec \n'.format(hours, minutes, seconds))

    def eval(self, dataloader=None, verbose=False):
        """
        Runs evaluation over the default evaluation dataloader or over a custom one if provided.
        """
        self.eval_epoch(dataloader=dataloader, verbose=verbose)

    def store_weights_and_energy(self, test_loader):
        """
        Reads the network's weights and, for each batch in test_loader, computes the equilibrium energy
        and per-neuron data from the network’s free layers. The results are then stored in a structured
        .hdf5 file in the results directory.
        """
        eq_neuron_dict = {}
        # Get the free layers; assume each has a 'name' attribute.
        hidden_layers = self._network.free_layers()
        hidden_layer_names = []
        for layer in hidden_layers:
            hidden_layer_names.append(layer.name)
            eq_neuron_dict[layer.name] = []

        eq_energies = []
        all_data = []

        for k, batch in enumerate(test_loader):
            # Unpack the batch (assumes either (x, y, idx) or (x, y))
            try:
                x, y, idx = batch
            except ValueError:
                x, y = batch
            all_data.append(x)
            self._network.set_input(x, reset=True)
            # Compute equilibrium energy and per-layer neuron data.
            en, n = self._energy_minimizer.compute_eq_energy()
            eq_energies.append(en)
            for layer_name in hidden_layer_names:
                layer_neurons = n[layer_name]
                eq_neuron_dict[layer_name].append(layer_neurons)

        # Stack tensors and convert to NumPy arrays.
        for layer_name in hidden_layer_names:
            neuron_list = eq_neuron_dict[layer_name]
            stacked_tensor = torch.cat(neuron_list, dim=0)
            eq_neuron_dict[layer_name] = stacked_tensor.cpu().numpy()

        eq_energies = torch.cat(eq_energies, dim=0).cpu().numpy()
        all_data = torch.cat(all_data, dim=0).cpu().numpy()

        # Also retrieve network weights.
        weights = {}
        state_dict = self._network.state_dict()
        for key, val in state_dict.items():
            weights[key] = val.cpu().numpy()

        # Write the weights, equilibrium energies, and neuron data to an HDF5 file.
        hdf5_path = os.path.join(self._config['result_path'], 'results.hdf5')
        with h5py.File(hdf5_path, 'w') as hf:
            # Create a group for weights.
            weights_group = hf.create_group('weights')
            for key, val in weights.items():
                weights_group.create_dataset(key, data=val)
            
            # Save equilibrium energies and all data.
            hf.create_dataset('eq_energies', data=eq_energies)
            hf.create_dataset('all_data', data=all_data)
            
            # Save the per-layer neuron data.
            eq_neuron_group = hf.create_group('eq_neuron_dict')
            for layer_name in hidden_layer_names:
                eq_neuron_group.create_dataset(layer_name, data=eq_neuron_dict[layer_name])
        
        print("Saved weights and equilibrium energy results to", hdf5_path)

    def __str__(self):
        return ('Epoch {}:\nTRAIN -- {}\nEVAL  -- {}'
                .format(self._epoch, self._stats_str(train=True), self._stats_str(train=False)))

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the network using NetworkRunner with extended features.')
    parser.add_argument('--config', '-c', type=str, default='config/config.json', help='Path to config JSON file.')
    parser.add_argument('--detection_model', '-dm', type=str, default='MLP', help='Type of detection model to use.')
    parser.add_argument('--main_model', '-mm', type=str, default='Llama2', help='Type of main model to use.')
    parser.add_argument('--dataset', '-ds', type=str, default='CIFAR10', help='Dataset to use for training.')
    parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode.')
    parser.add_argument('--wandb', type=bool, default=True, help='Enable Weights & Biases logging.')
    parser.add_argument('--log', type=bool, default=True, help='Enable logging to file.')
    parser.add_argument('--tag', type=str, default='', help='Tag for the wandb run.')
    parser.add_argument('--path', type=str, default=None, help='Path to save training outputs.')
    parser.add_argument('--use_tensorboard', type=bool, default=True, help='Enable TensorBoard logging.')

    args = parser.parse_args()

    # Load and set up configuration (directories, logging, wandb) using the first file's functions.
    config = set_config(args)

    # --- Dummy implementations for demonstration ---
    # Replace these with your actual network, estimator, scheduler, and dataloaders.
    class DummyNetwork:
        def __init__(self):
            self.params = []  # List of parameter objects
        def set_input(self, x, reset=True):
            pass
        def save(self, path):
            print("Saving network to", path)
        def layers(self):
            return []  # Return a list of layers
        def free_layers(self):
            # For demonstration, return dummy layer objects with a name attribute.
            DummyLayer = lambda name: type('DummyLayer', (), {'name': name})
            return [DummyLayer('layer1')(), DummyLayer('layer2')()]
        def state_dict(self):
            # Return a dummy state dict.
            return {'weight1': torch.randn(3, 3), 'bias1': torch.randn(3)}
    
    class DummyEstimator:
        def __init__(self):
            # Dummy cost function with a set_target method and params() method.
            self.cost_function = lambda: None
            self.cost_fn = type('CostFn', (), {'set_target': lambda self, y: None, 'params': lambda self: []})()
            # Dummy gradient estimator with compute_gradient method.
            self.gradient_estimator = type('GradEst', (), {'compute_gradient': lambda self: []})()
            # Dummy energy minimizer with compute_eq_energy method.
            # Here we simulate that it returns a tensor energy and a dict of neuron activations.
            self.energy_minimizer = type('EnergyMin', (), {
                'compute_equilibrium': lambda self: None,
                'compute_eq_energy': lambda self: (torch.randn(5, 1), {'layer1': torch.randn(5, 10), 'layer2': torch.randn(5, 20)})
            })()
    
    dummy_network = DummyNetwork()
    dummy_estimator = DummyEstimator()
    dummy_optimizer = torch.optim.Adam(dummy_network.state_dict().values(), lr=0.001)
    dummy_scheduler = torch.optim.lr_scheduler.StepLR(dummy_optimizer, step_size=1, gamma=0.95)

    # Create dummy datasets and dataloaders.
    from torch.utils.data import DataLoader, Dataset
    class DummyDataset(Dataset):
        def __init__(self, size=10):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            # Return dummy data (x, y) or (x, y, idx) as needed.
            return (torch.randn(5, 3), torch.randint(0, 2, (5,)), idx)
    
    train_dataset = DummyDataset(10)
    eval_dataset = DummyDataset(5)
    train_loader = DataLoader(train_dataset, batch_size=2)
    eval_loader = DataLoader(eval_dataset, batch_size=2)

    # Instantiate the NetworkRunner with the dummy components.
    runner = NetworkRunner(dummy_network, dummy_optimizer, dummy_estimator, dummy_scheduler,
                            train_loader, eval_loader, config)

    # Start training (using a small number of epochs for demonstration).
    runner.train(num_epochs=3, verbose=True)

    # After training, store the weights and equilibrium energy to an HDF5 file.
    runner.store_weights_and_energy(eval_loader)
