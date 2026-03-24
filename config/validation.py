def _resolve_strategy_name(cfg):
    strategy = cfg.STRATEGY_NAME
    if isinstance(strategy, cfg.StrategyName):
        return strategy

    supported = ", ".join(item.value for item in cfg.StrategyName)
    raise ValueError(f"STRATEGY_NAME must be one of: {supported}")


def _validate_global(cfg) -> None:
    if cfg.NUM_PARTITIONS <= 0:
        raise ValueError("NUM_PARTITIONS must be > 0")
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


def _validate_dp(cfg) -> None:
    if cfg.DP_NOISE_MULTIPLIER < 0:
        raise ValueError("DP_NOISE_MULTIPLIER must be >= 0")
    if cfg.DP_CLIPPING_NORM <= 0:
        raise ValueError("DP_CLIPPING_NORM must be > 0")
    if cfg.DP_NUM_SAMPLED_CLIENTS <= 0:
        raise ValueError("DP_NUM_SAMPLED_CLIENTS must be > 0")
    if cfg.DP_INITIAL_CLIPPING_NORM <= 0:
        raise ValueError("DP_INITIAL_CLIPPING_NORM must be > 0")
    if not (0 <= cfg.DP_TARGET_CLIPPED_QUANTILE <= 1):
        raise ValueError("DP_TARGET_CLIPPED_QUANTILE must be in [0, 1]")
    if cfg.DP_CLIP_NORM_LR <= 0:
        raise ValueError("DP_CLIP_NORM_LR must be > 0")
    if cfg.DP_CLIPPED_COUNT_STDDEV is not None and cfg.DP_CLIPPED_COUNT_STDDEV < 0:
        raise ValueError("DP_CLIPPED_COUNT_STDDEV must be >= 0 when provided")


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


def _validate_strategy_specific(cfg, strategy) -> None:
    if strategy == cfg.StrategyName.FEDAVGM:
        if cfg.SERVER_LEARNING_RATE <= 0:
            raise ValueError("SERVER_LEARNING_RATE must be > 0")
        if cfg.SERVER_MOMENTUM < 0:
            raise ValueError("SERVER_MOMENTUM must be >= 0")
        return

    if strategy == cfg.StrategyName.FEDPROX:
        if cfg.PROXIMAL_MU < 0:
            raise ValueError("PROXIMAL_MU must be >= 0")
        return

    if strategy == cfg.StrategyName.FEDADAGRAD:
        _validate_adaptive_common(cfg)
        return

    if strategy in {cfg.StrategyName.FEDADAM, cfg.StrategyName.FEDYOGI}:
        _validate_adaptive_common(cfg)
        _validate_betas(cfg)
        return

    if strategy == cfg.StrategyName.FEDTRIMMEDAVG:
        if not (0 <= cfg.FEDTRIMMEDAVG_BETA < 0.5):
            raise ValueError("FEDTRIMMEDAVG_BETA must be in [0, 0.5)")
        return

    if strategy == cfg.StrategyName.BULYAN:
        if cfg.BULYAN_NUM_MALICIOUS_NODES < 0:
            raise ValueError("BULYAN_NUM_MALICIOUS_NODES must be >= 0")
        return

    if strategy == cfg.StrategyName.KRUM:
        if cfg.KRUM_NUM_MALICIOUS_NODES < 0:
            raise ValueError("KRUM_NUM_MALICIOUS_NODES must be >= 0")
        return

    if strategy == cfg.StrategyName.MULTIKRUM:
        if cfg.MULTIKRUM_NUM_MALICIOUS_NODES < 0:
            raise ValueError("MULTIKRUM_NUM_MALICIOUS_NODES must be >= 0")
        if cfg.MULTIKRUM_NUM_NODES_TO_SELECT <= 0:
            raise ValueError("MULTIKRUM_NUM_NODES_TO_SELECT must be > 0")
        return

    if strategy in {
        cfg.StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING,
        cfg.StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING,
        cfg.StrategyName.DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING,
        cfg.StrategyName.DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING,
    }:
        _validate_dp(cfg)
        return

    if strategy == cfg.StrategyName.QFEDAVG:
        if cfg.QFEDAVG_CLIENT_LEARNING_RATE <= 0:
            raise ValueError("QFEDAVG_CLIENT_LEARNING_RATE must be > 0")
        if cfg.QFEDAVG_Q < 0:
            raise ValueError("QFEDAVG_Q must be >= 0")
        return

    if strategy in {cfg.StrategyName.FEDXGBBAGGING, cfg.StrategyName.FEDXGBCYCLIC}:
        _validate_xgb(cfg)
        return


def validate_config_module(cfg) -> None:
    """Validate config module values passed by caller to avoid import cycles."""
    strategy = _resolve_strategy_name(cfg)
    _validate_global(cfg)
    _validate_strategy_specific(cfg, strategy)
