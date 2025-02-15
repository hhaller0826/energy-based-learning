import torch
from ebm.util.config import Config
# from ebm.networks import get_network
from ebm.networks import DeepHopfieldNetwork
from ebm.estimators.gradient_estimators import EquiPropEstimator
from ebm.estimators.optimizer import SGDOptimizer
from ebm.estimators.cost import SquaredError
from ebm.runner import NetworkRunner

# Create a toy d
import torch
from ebm.util.config import Config
from ebm.networks import get_network
from ebm.networks import DeepHopfieldNetwork
from ebm.estimators.gradient_estimators import EquiPropEstimator
from ebm.estimators.optimizer import SGDOptimizer
from ebm.estimators.cost import SquaredError
from ebm.runner import NetworkRunner

# Create a toy dataset of 2 interlocking rings in 3D and with 1000 points and labels for each point
# Create a toy dataset of interlocking rings perpendicular to each other in 3D
num_points = 500
theta = torch.linspace(0, 2 * torch.pi, num_points)

# First ring in the XY plane, centered at (1, 0, 0)
x1 = 1 + torch.cos(theta)
y1 = torch.sin(theta)
z1 = torch.zeros(num_points)

# Second ring in the XZ plane, centered at (0, 0, 1)
x2 = torch.cos(theta)
y2 = torch.zeros(num_points)
z2 = 1 + torch.sin(theta)

# Combine the rings
X1 = torch.stack([x1, y1, z1], dim=1)
X2 = torch.stack([x2, y2, z2], dim=1)
X = torch.cat([X1, X2], dim=0)

# Labels for the rings
Y1 = torch.zeros(num_points, dtype=torch.int32)
Y2 = torch.ones(num_points, dtype=torch.int32)
Y = torch.cat([Y1, Y2], dim=0)

# Create a dataloader for the dataset Y is long tensor
Y = Y.long()
dataset = torch.utils.data.TensorDataset(X, Y)

# Split the dataset into a training and test set
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# Create a network, cost function, optimizer, estimator, and runner
config = Config()
layers = [3,128,2] 
network = DeepHopfieldNetwork(layers,config)
cost_function = SquaredError(network)
optimizer = SGDOptimizer(network,cost_function)
estimator = EquiPropEstimator(network, cost_function)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
runner = NetworkRunner(network,estimator,optimizer,scheduler, train_loader, test_loader)
runner.train(num_epochs=100, verbose=True)
runner.eval()