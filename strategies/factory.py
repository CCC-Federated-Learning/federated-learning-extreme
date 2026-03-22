import config

from .fedavg import build_fedavg
from .fedavgm import build_fedavgm
from .fedadagrad import build_fedadagrad
from .fedadam import build_fedadam
from .fedprox import build_fedprox
from .fedyogi import build_fedyogi
from .bulyan import build_bulyan
from .krum import build_krum
from .multikrum import build_multikrum
from .fedmedian import build_fedmedian
from .fedtrimmedavg import build_fedtrimmedavg
from .dp_client_adaptive import build_dp_client_adaptive
from .dp_client_fixed import build_dp_client_fixed
from .dp_server_adaptive import build_dp_server_adaptive
from .dp_server_fixed import build_dp_server_fixed
from .fedxgbbagging import build_fedxgbbagging
from .fedxgbcyclic import build_fedxgbcyclic
from .qfedavg import build_qfedavg


# Strategy registry: add more built-in Flower strategies here.
STRATEGY_REGISTRY = {
    "fedavg": build_fedavg,
    "fedavgm": build_fedavgm,
    "fedadagrad": build_fedadagrad,
    "fedadam": build_fedadam,
    "fedprox": build_fedprox,
    "fedyogi": build_fedyogi,
    "bulyan": build_bulyan,
    "krum": build_krum,
    "multikrum": build_multikrum,
    "fedmedian": build_fedmedian,
    "fedtrimmedavg": build_fedtrimmedavg,
    "differentialprivacyclientsideadaptiveclipping": build_dp_client_adaptive,
    "differentialprivacyclientsidefixedclipping": build_dp_client_fixed,
    "differentialprivacyserversideadaptiveclipping": build_dp_server_adaptive,
    "differentialprivacyserversidefixedclipping": build_dp_server_fixed,
    "fedxgbbagging": build_fedxgbbagging,
    "fedxgbcyclic": build_fedxgbcyclic,
    "qfedavg": build_qfedavg,
}


def build_strategy():
    strategy_key = config.STRATEGY_NAME.lower()
    if strategy_key not in STRATEGY_REGISTRY:
        supported = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported STRATEGY_NAME: {config.STRATEGY_NAME}. Supported: {supported}"
        )
    return STRATEGY_REGISTRY[strategy_key]()
