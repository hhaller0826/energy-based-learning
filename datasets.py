import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

class TwoMoonsDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        A PyTorch Dataset for the Two Moons data.
        
        Args:
            data (array-like): Input data points (features).
            labels (array-like): Target labels.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample, label = self.data[idx], self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

class IndexedDataset(torch.utils.data.Dataset):
    """
    Wrapper class that turns a (labeled) datadet into an `indexed dataset'.

    Each item of an `indexed dataset' consists of an image, its label, and its index in the dataset.
    In contrast, the standard Dataset class does not return the index of the image

    Attributes
    ----------
    _dataset (Dataset): the original dataset that we wrap into an `indexed dataset'
    """

    def __init__(self, dataset):
        """Creates an instance of IndexedDataset

        Args:
            dataset (Dataset): the original dataset that we wrap into an `indexed dataset'
        """
        self._dataset = dataset
        
    def __getitem__(self, index):
        """Returns the image, its label, and its index."""
        data, target = self._dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self._dataset)

def load_two_moons(n_samples=1000, noise=0.1, test_size=0.2, transform=None):
    """
    Loads the Two Moons dataset with train/test splits.

    Args:
        n_samples (int): Total number of samples to generate.
        noise (float): Noise level for the dataset.
        test_size (float): Proportion of the dataset to be used as test data.
        transform (callable, optional): Optional transform to apply to the data.

    Returns:
        tuple: Training dataset and test dataset as PyTorch Dataset objects.
    """
    # Generate two moons dataset
    data, labels = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # Split into training and test sets
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=test_size, random_state=42
    )

    # Create PyTorch Datasets
    training_data = TwoMoonsDataset(data_train, labels_train, transform=transform)
    test_data = TwoMoonsDataset(data_test, labels_test, transform=transform)

    return training_data, test_data

def load_mnist(normalize=False, augment_32x32=False):
    """Loads the MNIST dataset (training and test sets).

    Args:
        normalize (bool, optional): Whether to normalize the data or not. Default is False.
        augment_32x32 (bool, optional): Whether data images are augmented to 32x32 pixels, or left to 28x28 pixels. Default is False.

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """
    
    training_transforms = list()
    test_transforms = list()

    if augment_32x32:
        training_transforms.append(torchvision.transforms.Pad(padding=(2, 2, 2, 2), fill=0))  # Add 2 pixels of padding on each side
        training_transforms.append(torchvision.transforms.RandomCrop(size=[32,32], padding=2, padding_mode='edge'))

        test_transforms.append(torchvision.transforms.Pad(padding=(2, 2, 2, 2), fill=0))  # Add 2 pixels of padding on each side
    
    training_transforms.append(torchvision.transforms.ToTensor())
    test_transforms.append(torchvision.transforms.ToTensor())

    if normalize:
        mean = (0.1307,)
        std = (0.3081,)
        training_transforms.append(torchvision.transforms.Normalize(mean, std))
        test_transforms.append(torchvision.transforms.Normalize(mean, std))

    training_data = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(training_transforms),
    )

    test_data = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(test_transforms),
    )

    return training_data, test_data


def load_fashion_mnist(normalize=True, augment_32x32=False):
    """Loads the FashionMNIST dataset (training and test sets).

    Args:
        normalize (bool, optional): Whether to normalize the data or not. Default is True.
        augment_32x32 (bool, optional): Whether data images are augmented to 32x32 pixels, or left to 28x28 pixels. Default is False.

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """
    
    training_transforms = list()
    test_transforms = list()

    if augment_32x32:
        training_transforms.append(torchvision.transforms.Pad(padding=(2, 2, 2, 2), fill=0))  # Add 2 pixels of padding on each side
        training_transforms.append(torchvision.transforms.RandomCrop(size=[32,32], padding=2, padding_mode='edge'))

        test_transforms.append(torchvision.transforms.Pad(padding=(2, 2, 2, 2), fill=0))  # Add 2 pixels of padding on each side
    
    training_transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    training_transforms.append(torchvision.transforms.ToTensor())
    test_transforms.append(torchvision.transforms.ToTensor())

    if normalize:
        mean = (0.2860,)
        std = (0.3530,)
        training_transforms.append(torchvision.transforms.Normalize(mean, std))
        test_transforms.append(torchvision.transforms.Normalize(mean, std))

    training_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(training_transforms),
    )

    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(test_transforms),
    )

    return training_data, test_data


def load_svhn(normalize=True):
    """Loads the SVHN dataset (training and test sets).

    Args:
        normalize (bool, optional): Whether to normalize the data or not. Default is True.

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """

    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    if normalize:
        final_transform = torchvision.transforms.Normalize(mean, std)
    else:
        final_transform = torchvision.transforms.Lambda(lambda x: x)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
        torchvision.transforms.ToTensor(),
        final_transform
    ])

    # Download training data
    training_data = torchvision.datasets.SVHN(
        root="data",
        split='train',
        download=True,
        transform=train_transform,
    )

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        final_transform
    ])

    # Download test data
    test_data = torchvision.datasets.SVHN(
        root="data",
        split='test',
        download=True,
        transform=test_transform,
    )

    training_data.classes = classes
    test_data.classes = classes

    return training_data, test_data


def load_cifar10(normalize=True):
    """Loads the CIFAR10 dataset (training and test sets).

    Args:
        normalize (bool, optional): Whether to normalize the data or not. Default is True.

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """

    mean=(0.4914, 0.4822, 0.4465)
    std=(3*0.2023, 3*0.1994, 3*0.2010)

    if normalize:
        final_transform = torchvision.transforms.Normalize(mean, std)
    else:
        final_transform = torchvision.transforms.Lambda(lambda x: x)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
        torchvision.transforms.ToTensor(),
        final_transform
    ])

    # Download training data
    training_data = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        final_transform
    ])

    # Download test data
    test_data = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )

    return training_data, test_data


def load_cifar100(normalize=True):
    """Loads the CIFAR100 dataset (training and test sets).

    Args:
        normalize (bool, optional): Whether to normalize the data or not. Default is True.

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    if normalize:
        final_transform = torchvision.transforms.Normalize(mean, std)
    else:
        final_transform = torchvision.transforms.Lambda(lambda x: x)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),
        torchvision.transforms.ToTensor(),
        final_transform
    ])

    # Download training data
    training_data = torchvision.datasets.CIFAR100(
        root="data",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        final_transform
    ])

    # Download test data
    test_data = torchvision.datasets.CIFAR100(
        root="data",
        train=False,
        download=True,
        transform=test_transform,
    )

    return training_data, test_data


def load_dataset(dataset, normalize=True, augment_32x32=False):
    """Loads the dataset (training and test sets).

    Args:
        dataset (str): The dataset used for training. Either 'MNIST', 'FashionMNIST', 'CIFAR10' or 'CIFAR100'.
        normalize (bool, optional): raw data if False, or pre-processed (e.g. normalized) data if True. Default: True
        augment_32x32 (bool, optional): if True, and if dataset is MNIST or Fashion-MNIST, augment the input images to 32x32 pixels. Default: False

    Returns:
        tuple of datasets: the training dataset and the test dataset
    """

    if dataset == 'MNIST': return load_mnist(normalize, augment_32x32)
    elif dataset == 'FashionMNIST': return load_fashion_mnist(normalize, augment_32x32)
    elif dataset == 'SVHN': return load_svhn(normalize)
    elif dataset == 'CIFAR10': return load_cifar10(normalize)
    elif dataset == 'CIFAR100': return load_cifar100(normalize)
    elif dataset == 'TwoMoons': return load_two_moons()
    else: raise ValueError("expected 'MNIST', 'FashionMNIST', `SVHN', `CIFAR10', `CIFAR100', or 'TwoMoons' but got {}".format(dataset))

def load_dataloaders(dataset, batch_size, augment_32x32=False, normalize=True):
    """Builds data loaders (training and test loaders).

    The test loader is an IndexedDataset, meaning that it generates data in the form of triplets (x, y, idx).

    Args:
        dataset (str): The dataset used for training. Either 'MNIST' or 'FashionMNIST'
        batch_size (int): size of the mini-batch
        augment_32x32 (bool, optional): if True, and if dataset is MNIST or Fashion-MNIST, augment the input images to 32x32 pixels. Default: False
        normalize (bool, optional): raw data if False, or pre-processed (e.g. normalized) data if True. Default: True

    Returns:
        tuple of dataloaders: the training loader and the test loader
    """

    training_data, test_data = load_dataset(dataset, normalize=normalize, augment_32x32=augment_32x32)

    training_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_data = IndexedDataset(test_data)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=1)

    return training_loader, test_loader