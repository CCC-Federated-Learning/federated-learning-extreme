from enum import Enum


class _ConfigEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class StrategyName(_ConfigEnum):
    # Basic & Classic Strategies
    FEDAVG = "FedAvg"
    FEDAVGM = "FedAvgM"
    FEDPROX = "FedProx"

    # Adaptive Strategies
    FEDADAGRAD = "FedAdagrad"
    FEDADAM = "FedAdam"
    FEDYOGI = "FedYogi"

    # Robust Aggregation Strategies
    BULYAN = "Bulyan"
    KRUM = "Krum"
    MULTIKRUM = "MultiKrum"
    FEDMEDIAN = "FedMedian"
    FEDTRIMMEDAVG = "FedTrimmedAvg"

    # Differential Privacy Strategies
    DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING = "DifferentialPrivacyClientSideAdaptiveClipping"
    DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING = "DifferentialPrivacyClientSideFixedClipping"
    DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING = "DifferentialPrivacyServerSideAdaptiveClipping"
    DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING = "DifferentialPrivacyServerSideFixedClipping"

    # XGBoost specific strategies
    FEDXGBBAGGING = "FedXgbBagging"
    FEDXGBCYCLIC = "FedXgbCyclic"

    # Fairness and other strategies
    QFEDAVG = "QFedAvg"


class DataDistribution(_ConfigEnum):
    IID = "IID"
    DIRICHLET = "Dirichlet"
    LABEL = "label"


class DatasetName(_ConfigEnum):
    MNIST = "MNIST"
