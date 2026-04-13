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


# Strategy registry mapped by StrategyName enum
STRATEGY_REGISTRY = {
    "FEDAVG": build_fedavg,
    "FEDAVGM": build_fedavgm,
    "FEDADAGRAD": build_fedadagrad,
    "FEDADAM": build_fedadam,
    "FEDPROX": build_fedprox,
    "FEDYOGI": build_fedyogi,
    "BULYAN": build_bulyan,
    "KRUM": build_krum,
    "MULTIKRUM": build_multikrum,
    "FEDMEDIAN": build_fedmedian,
    "FEDTRIMMEDAVG": build_fedtrimmedavg,
    "DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING": build_dp_client_adaptive,
    "DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING": build_dp_client_fixed,
    "DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING": build_dp_server_adaptive,
    "DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING": build_dp_server_fixed,
    "FEDXGBBAGGING": build_fedxgbbagging,
    "FEDXGBCYCLIC": build_fedxgbcyclic,
    "QFEDAVG": build_qfedavg,
}


def build_strategy():
    # Use enum.name (uppercase) directly - no string manipulation needed
    strategy_name = config.STRATEGY_NAME.name
    if strategy_name not in STRATEGY_REGISTRY:
        supported = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported STRATEGY_NAME: {config.STRATEGY_NAME}. Supported: {supported}"
        )
    return STRATEGY_REGISTRY[strategy_name]()
