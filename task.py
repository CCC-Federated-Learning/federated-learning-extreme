import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """輕量級 CNN for MNIST (28x28 grayscale)"""
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 28x28 -> pool -> 14x14 -> pool -> 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

_cached_train = None
_cached_test = None


def _get_datasets():
    """Load and cache MNIST train/test datasets."""
    global _cached_train, _cached_test
    if _cached_train is None:
        _cached_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=pytorch_transforms
        )
        _cached_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=pytorch_transforms
        )
    return _cached_train, _cached_test


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition MNIST data — each client only gets samples of its own label."""
    train_dataset, test_dataset = _get_datasets()

    # Filter by label: each client only gets samples matching its partition_id
    train_indices = (train_dataset.targets == partition_id).nonzero(as_tuple=True)[0].tolist()
    test_indices = (test_dataset.targets == partition_id).nonzero(as_tuple=True)[0].tolist()

    trainloader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(
        Subset(test_dataset, test_indices), batch_size=batch_size
    )
    return trainloader, testloader


def load_centralized_dataset():
    """Load full test set and return dataloader."""
    _, test_dataset = _get_datasets()
    return DataLoader(test_dataset, batch_size=128)


def train_fn(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test_fn(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
