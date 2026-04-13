import math


def _resolve_strategy(cfg):
    strategy = cfg.STRATEGY_NAME
    if isinstance(strategy, cfg.StrategyName):
        return strategy
    supported = ", ".join(item.value for item in cfg.StrategyName)
    raise ValueError(f"STRATEGY_NAME must be one of: {supported}")


def _validate_global(cfg) -> None:
    if cfg.NUM_CLIENTS <= 0:
        raise ValueError("NUM_CLIENTS must be > 0")
    if cfg.BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0")
    if cfg.LOCAL_EPOCHS <= 0:
        raise ValueError("LOCAL_EPOCHS must be > 0")
    if cfg.NUM_ROUNDS <= 0:
        raise ValueError("NUM_ROUNDS must be > 0")
    if cfg.LR <= 0:
        raise ValueError("LR must be > 0")
    if not isinstance(cfg.DATA_DISTRIBUTION, cfg.DataDistribution):
        supported = ", ".join(item.value for item in cfg.DataDistribution)
        raise ValueError(f"DATA_DISTRIBUTION must be one of: {supported}")
    if not isinstance(cfg.DATASET_NAME, cfg.DatasetName):
        supported = ", ".join(item.value for item in cfg.DatasetName)
        raise ValueError(f"DATASET_NAME must be one of: {supported}")
    if cfg.DIRICHLET_ALPHA <= 0:
        raise ValueError("DIRICHLET_ALPHA must be > 0")
    if not (0 < cfg.FRACTION_TRAIN <= 1):
        raise ValueError("FRACTION_TRAIN must be in (0, 1]")
    if not (0 <= cfg.FRACTION_EVALUATE <= 1):
        raise ValueError("FRACTION_EVALUATE must be in [0, 1]")


def _validate_adaptive_common(cfg) -> None:
    if cfg.SERVER_ETA <= 0:
        raise ValueError("SERVER_ETA must be > 0")
    if cfg.SERVER_ETA_L <= 0:
        raise ValueError("SERVER_ETA_L must be > 0")
    if cfg.SERVER_TAU <= 0:
        raise ValueError("SERVER_TAU must be > 0")


def _validate_betas(cfg) -> None:
    if not (0 < cfg.BETA_1 < 1):
        raise ValueError("BETA_1 must be in (0, 1)")
    if not (0 < cfg.BETA_2 < 1):
        raise ValueError("BETA_2 must be in (0, 1)")


def _estimated_train_clients(cfg) -> int:
    return max(2, int(cfg.NUM_CLIENTS * cfg.FRACTION_TRAIN))


def _validate_fedavgm(cfg) -> None:
    if cfg.SERVER_LEARNING_RATE <= 0:
        raise ValueError("SERVER_LEARNING_RATE must be > 0")
    if cfg.SERVER_MOMENTUM < 0:
        raise ValueError("SERVER_MOMENTUM must be >= 0")


def _validate_fedprox(cfg) -> None:
    if cfg.PROXIMAL_MU < 0:
        raise ValueError("PROXIMAL_MU must be >= 0")


def _validate_fedadagrad(cfg) -> None:
    _validate_adaptive_common(cfg)


def _validate_fedadam(cfg) -> None:
    _validate_adaptive_common(cfg)
    _validate_betas(cfg)


def _validate_fedyogi(cfg) -> None:
    _validate_adaptive_common(cfg)
    _validate_betas(cfg)


def _validate_fedtrimmedavg(cfg) -> None:
    if not (0 <= cfg.FEDTRIMMEDAVG_BETA < 0.5):
        raise ValueError("FEDTRIMMEDAVG_BETA must be in [0, 0.5)")


def _validate_bulyan(cfg) -> None:
    if cfg.BULYAN_NUM_MALICIOUS_NODES < 0:
        raise ValueError("BULYAN_NUM_MALICIOUS_NODES must be >= 0")
    n = _estimated_train_clients(cfg)
    f = cfg.BULYAN_NUM_MALICIOUS_NODES
    if n < 4 * f + 3:
        raise ValueError(
            f"Bulyan requires n >= 4f+3. Got n={n}, f={f}. "
            "Increase FRACTION_TRAIN/NUM_CLIENTS or reduce BULYAN_NUM_MALICIOUS_NODES."
        )


def _validate_krum(cfg) -> None:
    if cfg.KRUM_NUM_MALICIOUS_NODES < 0:
        raise ValueError("KRUM_NUM_MALICIOUS_NODES must be >= 0")
    n = _estimated_train_clients(cfg)
    f = cfg.KRUM_NUM_MALICIOUS_NODES
    if n < 2 * f + 3:
        raise ValueError(
            f"Krum requires n >= 2f+3. Got n={n}, f={f}. "
            "Increase FRACTION_TRAIN/NUM_CLIENTS or reduce KRUM_NUM_MALICIOUS_NODES."
        )


def _validate_multikrum(cfg) -> None:
    if cfg.MULTIKRUM_NUM_MALICIOUS_NODES < 0:
        raise ValueError("MULTIKRUM_NUM_MALICIOUS_NODES must be >= 0")
    if cfg.MULTIKRUM_NUM_NODES_TO_SELECT <= 0:
        raise ValueError("MULTIKRUM_NUM_NODES_TO_SELECT must be > 0")
    n = _estimated_train_clients(cfg)
    f = cfg.MULTIKRUM_NUM_MALICIOUS_NODES
    m = cfg.MULTIKRUM_NUM_NODES_TO_SELECT
    if n < 2 * f + 3:
        raise ValueError(
            f"MultiKrum requires n >= 2f+3. Got n={n}, f={f}. "
            "Increase FRACTION_TRAIN/NUM_CLIENTS or reduce MULTIKRUM_NUM_MALICIOUS_NODES."
        )
    limit = n - f - 2
    if m > limit:
        raise ValueError(
            f"MULTIKRUM_NUM_NODES_TO_SELECT too large. Need m <= n-f-2={limit}, got m={m}."
        )


def _validate_dp(cfg) -> None:
    if cfg.DP_NOISE_MULTIPLIER < 0:
        raise ValueError("DP_NOISE_MULTIPLIER must be >= 0")
    if cfg.DP_CLIPPING_NORM <= 0:
        raise ValueError("DP_CLIPPING_NORM must be > 0")
    if cfg.DP_NUM_SAMPLED_CLIENTS <= 0:
        raise ValueError("DP_NUM_SAMPLED_CLIENTS must be > 0")
    n = _estimated_train_clients(cfg)
    if cfg.DP_NUM_SAMPLED_CLIENTS > n:
        raise ValueError(
            f"DP_NUM_SAMPLED_CLIENTS ({cfg.DP_NUM_SAMPLED_CLIENTS}) exceeds "
            f"estimated train clients per round ({n})."
        )
    if cfg.DP_INITIAL_CLIPPING_NORM <= 0:
        raise ValueError("DP_INITIAL_CLIPPING_NORM must be > 0")
    if not (0 <= cfg.DP_TARGET_CLIPPED_QUANTILE <= 1):
        raise ValueError("DP_TARGET_CLIPPED_QUANTILE must be in [0, 1]")
    if cfg.DP_CLIP_NORM_LR <= 0:
        raise ValueError("DP_CLIP_NORM_LR must be > 0")
    if cfg.DP_CLIPPED_COUNT_STDDEV is not None and cfg.DP_CLIPPED_COUNT_STDDEV < 0:
        raise ValueError("DP_CLIPPED_COUNT_STDDEV must be >= 0 when set")


def _validate_qfedavg(cfg) -> None:
    if not math.isfinite(cfg.QFEDAVG_CLIENT_LEARNING_RATE) or cfg.QFEDAVG_CLIENT_LEARNING_RATE <= 0:
        raise ValueError("QFEDAVG_CLIENT_LEARNING_RATE must be a finite positive number")
    if not math.isfinite(cfg.QFEDAVG_Q) or cfg.QFEDAVG_Q < 0:
        raise ValueError("QFEDAVG_Q must be a finite non-negative number")


def _validate_xgb(cfg) -> None:
    if cfg.XGB_NUM_LOCAL_ROUND <= 0:
        raise ValueError("XGB_NUM_LOCAL_ROUND must be > 0")
    if cfg.XGB_MAX_DEPTH <= 0:
        raise ValueError("XGB_MAX_DEPTH must be > 0")
    if cfg.XGB_ETA <= 0:
        raise ValueError("XGB_ETA must be > 0")
    if not (0 < cfg.XGB_SUBSAMPLE <= 1):
        raise ValueError("XGB_SUBSAMPLE must be in (0, 1]")
    if not (0 < cfg.XGB_COLSAMPLE_BYTREE <= 1):
        raise ValueError("XGB_COLSAMPLE_BYTREE must be in (0, 1]")
    if cfg.XGB_MIN_CHILD_WEIGHT < 0:
        raise ValueError("XGB_MIN_CHILD_WEIGHT must be >= 0")
    if cfg.XGB_REG_LAMBDA < 0:
        raise ValueError("XGB_REG_LAMBDA must be >= 0")
    if cfg.XGB_NUM_CLASS <= 1:
        raise ValueError("XGB_NUM_CLASS must be > 1")


_STRATEGY_VALIDATORS = None


def _get_validators(cfg) -> dict:
    return {
        cfg.StrategyName.FEDAVG:     lambda _: None,
        cfg.StrategyName.FEDAVGM:    _validate_fedavgm,
        cfg.StrategyName.FEDPROX:    _validate_fedprox,
        cfg.StrategyName.FEDADAGRAD: _validate_fedadagrad,
        cfg.StrategyName.FEDADAM:    _validate_fedadam,
        cfg.StrategyName.FEDYOGI:    _validate_fedyogi,
        cfg.StrategyName.FEDMEDIAN:  lambda _: None,
        cfg.StrategyName.FEDTRIMMEDAVG: _validate_fedtrimmedavg,
        cfg.StrategyName.BULYAN:     _validate_bulyan,
        cfg.StrategyName.KRUM:       _validate_krum,
        cfg.StrategyName.MULTIKRUM:  _validate_multikrum,
        cfg.StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING: _validate_dp,
        cfg.StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING:    _validate_dp,
        cfg.StrategyName.DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING: _validate_dp,
        cfg.StrategyName.DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING:    _validate_dp,
        cfg.StrategyName.QFEDAVG:       _validate_qfedavg,
        cfg.StrategyName.FEDXGBBAGGING: _validate_xgb,
        cfg.StrategyName.FEDXGBCYCLIC:  _validate_xgb,
    }


def validate_config_module(cfg) -> None:
    strategy = _resolve_strategy(cfg)
    _validate_global(cfg)
    validators = _get_validators(cfg)
    validator = validators.get(strategy)
    if validator is None:
        raise ValueError(f"No validator for strategy: {strategy}")
    validator(cfg)
