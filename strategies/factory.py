import config

from .fedavg import build_fedavg
from .fedavgm import build_fedavgm
from .fedprox import build_fedprox


# Strategy registry: add more built-in Flower strategies here.
STRATEGY_REGISTRY = {
    "fedavg": build_fedavg,
    "fedavgm": build_fedavgm,
    "fedprox": build_fedprox,
}


def build_strategy():
    strategy_key = config.STRATEGY_NAME.lower()
    if strategy_key not in STRATEGY_REGISTRY:
        supported = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported STRATEGY_NAME: {config.STRATEGY_NAME}. Supported: {supported}"
        )
    return STRATEGY_REGISTRY[strategy_key]()
