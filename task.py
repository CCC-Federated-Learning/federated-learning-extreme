import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
_partition_cache = {"train": {}, "test": {}}


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


def _build_partition_indices(
    targets: torch.Tensor,
    num_partitions: int,
    distribution: str,
    dirichlet_alpha: float,
    seed: int,
):
    """Build sample indices for each client partition."""
    distribution = distribution.lower()

    if distribution == "label":
        return [
            (targets == (partition_idx % 10)).nonzero(as_tuple=True)[0].tolist()
            for partition_idx in range(num_partitions)
        ]

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(targets))

    if distribution == "iid":
        rng.shuffle(all_indices)
        return [indices.tolist() for indices in np.array_split(all_indices, num_partitions)]

    if distribution == "dirichlet":
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0 when using dirichlet distribution")

        targets_np = targets.cpu().numpy()
        client_indices = [[] for _ in range(num_partitions)]

        for label in np.unique(targets_np):
            label_indices = np.where(targets_np == label)[0]
            rng.shuffle(label_indices)

            proportions = rng.dirichlet(np.full(num_partitions, dirichlet_alpha))
            split_points = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
            splits = np.split(label_indices, split_points)

            for partition_idx, split in enumerate(splits):
                client_indices[partition_idx].extend(split.tolist())

        for partition_idx in range(num_partitions):
            rng.shuffle(client_indices[partition_idx])

        return client_indices

    raise ValueError(
        "distribution must be one of: 'iid', 'dirichlet', or 'label'"
    )


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    distribution: str = "iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
):
    """Load partitioned MNIST data with Flower-style data distributions.

    distribution options:
    - "iid": random even split across clients (Flower common baseline)
    - "dirichlet": non-IID split controlled by dirichlet_alpha
    - "label": legacy mode, each client mainly maps to one label
    """
    train_dataset, test_dataset = _get_datasets()
    distribution = distribution.lower()

    train_cache_key = (distribution, num_partitions, dirichlet_alpha, seed)
    test_cache_key = (distribution, num_partitions, dirichlet_alpha, seed)

    if train_cache_key not in _partition_cache["train"]:
        _partition_cache["train"][train_cache_key] = _build_partition_indices(
            train_dataset.targets,
            num_partitions,
            distribution,
            dirichlet_alpha,
            seed,
        )
    if test_cache_key not in _partition_cache["test"]:
        _partition_cache["test"][test_cache_key] = _build_partition_indices(
            test_dataset.targets,
            num_partitions,
            distribution,
            dirichlet_alpha,
            seed,
        )

    train_partitions = _partition_cache["train"][train_cache_key]
    test_partitions = _partition_cache["test"][test_cache_key]

    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id {partition_id} is out of range for num_partitions={num_partitions}"
        )

    train_indices = train_partitions[partition_id]
    test_indices = test_partitions[partition_id]

    # RandomSampler (shuffle=True) cannot be built on an empty dataset.
    train_shuffle = len(train_indices) > 0

    trainloader = DataLoader(
        Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=train_shuffle
    )
    testloader = DataLoader(
        Subset(test_dataset, test_indices), batch_size=batch_size
    )
    return trainloader, testloader


def load_centralized_dataset():
    """Load full test set and return dataloader."""
    _, test_dataset = _get_datasets()
    return DataLoader(test_dataset, batch_size=128)


def train_fn(
    net,
    trainloader,
    epochs,
    lr,
    device,
    proximal_mu: float = 0.0,
    global_params=None,
):
    """Train the model on the training set."""
    if len(trainloader) == 0:
        return 0.0

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

            if proximal_mu > 0 and global_params is not None:
                prox_term = 0.0
                for param, global_param in zip(net.parameters(), global_params):
                    prox_term += torch.sum((param - global_param) ** 2)
                loss = loss + 0.5 * proximal_mu * prox_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test_fn(net, testloader, device):
    """Validate the model on the test set."""
    if len(testloader) == 0 or len(testloader.dataset) == 0:
        return 0.0, 0.0

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
