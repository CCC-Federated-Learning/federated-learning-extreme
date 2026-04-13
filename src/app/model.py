import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset

from config import DataDistribution
from app.data_cache import get_datasets, get_partition_cache


class Net(nn.Module):
    """Lightweight CNN for MNIST (28×28 grayscale)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 28×28 → 14×14 → 7×7
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _build_partitions(
    targets: torch.Tensor,
    num_clients: int,
    distribution: DataDistribution,
    dirichlet_alpha: float,
    seed: int,
) -> list[list[int]]:
    """Return a list of per-client sample index lists for the given distribution."""
    if distribution == DataDistribution.LABEL:
        return [
            (targets == (i % 10)).nonzero(as_tuple=True)[0].tolist()
            for i in range(num_clients)
        ]

    rng = np.random.default_rng(seed)
    all_idx = np.arange(len(targets))

    if distribution == DataDistribution.IID:
        rng.shuffle(all_idx)
        return [chunk.tolist() for chunk in np.array_split(all_idx, num_clients)]

    if distribution == DataDistribution.DIRICHLET:
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0 for Dirichlet distribution")

        targets_np = targets.cpu().numpy()
        client_idx: list[list[int]] = [[] for _ in range(num_clients)]

        for label in np.unique(targets_np):
            label_pos = np.where(targets_np == label)[0]
            rng.shuffle(label_pos)
            proportions = rng.dirichlet(np.full(num_clients, dirichlet_alpha))
            split_points = (np.cumsum(proportions) * len(label_pos)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(label_pos, split_points)):
                client_idx[i].extend(chunk.tolist())

        for idx in client_idx:
            rng.shuffle(idx)

        return client_idx

    raise ValueError(
        f"Unknown distribution: {distribution}. "
        f"Choices: {', '.join(d.value for d in DataDistribution)}"
    )


def load_client_data(
    client_id: int,
    num_clients: int,
    batch_size: int,
    distribution: DataDistribution = DataDistribution.IID,
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Return (trainloader, testloader) for the given client."""
    if not (0 <= client_id < num_clients):
        raise ValueError(f"client_id {client_id} out of range [0, {num_clients})")

    train_ds, test_ds = get_datasets()
    cache = get_partition_cache()
    key = (distribution.name, num_clients, dirichlet_alpha, seed)

    train_parts = cache.get_or_build_train(
        key,
        lambda: _build_partitions(train_ds.targets, num_clients, distribution, dirichlet_alpha, seed),
    )
    test_parts = cache.get_or_build_test(
        key,
        lambda: _build_partitions(test_ds.targets, num_clients, distribution, dirichlet_alpha, seed),
    )

    train_idx = train_parts[client_id]
    test_idx = test_parts[client_id]

    trainloader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=len(train_idx) > 0,
    )
    testloader = DataLoader(Subset(test_ds, test_idx), batch_size=batch_size)
    return trainloader, testloader


def load_test_dataloader(batch_size: int = 128) -> DataLoader:
    """Return a DataLoader over the full MNIST test set for centralised evaluation."""
    _, test_ds = get_datasets()
    return DataLoader(test_ds, batch_size=batch_size)


def train_model(
    net: Net,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    proximal_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
) -> float:
    """Train `net` for `epochs` epochs; return average training loss."""
    if len(trainloader) == 0:
        return 0.0

    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()

    total_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)

            if proximal_mu > 0 and global_params is not None:
                prox = sum(
                    torch.sum((p - gp) ** 2)
                    for p, gp in zip(net.parameters(), global_params)
                )
                loss = loss + 0.5 * proximal_mu * prox

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / (epochs * len(trainloader))


def eval_model(
    net: Net,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate `net`; return (loss, accuracy)."""
    if len(dataloader) == 0 or len(dataloader.dataset) == 0:
        return 0.0, 0.0

    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return loss, accuracy
