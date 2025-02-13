from jaynes.network import JaynesNetwork
from resistive.network import DeepResistiveEnergy
from model.function.network import Network
from model.hopfield.network import DeepHopfieldEnergy
def get_model(config,energy_fn=None):
    """
    Get the model

    Returns:
        model: the model
    """
    model_name = config['model_name']
    if model_name == 'jaynes':
        energy_fn = JaynesNetwork(config)
        model = Network(energy_fn)
    elif model_name == 'resistive':
        energy_fn = DeepResistiveEnergy(config)
        model = Network(energy_fn)
    elif model_name == 'hopfield':
        energy_fn = DeepHopfieldEnergy(config)
        model = Network(energy_fn)
    elif energy_fn is not None:
        model = Network(energy_fn)
    else:
        raise ValueError("Model not found")
    return model
