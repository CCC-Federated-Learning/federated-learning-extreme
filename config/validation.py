import math


def _resolve_strategy_name(cfg):
    """Resolve and validate strategy name."""
    strategy = cfg.STRATEGY_NAME
    if isinstance(strategy, cfg.StrategyName):
        return strategy

    supported = ", ".join(item.value for item in cfg.StrategyName)
    raise ValueError(f"STRATEGY_NAME must be one of: {supported}")


def _validate_global(cfg) -> None:
    """Validate global configuration parameters."""
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
    """Validate parameters common to adaptive strategies (AdaGrad, Adam, YoGi)."""
    if cfg.SERVER_ETA <= 0:
        raise ValueError("SERVER_ETA must be > 0")
    if cfg.SERVER_ETA_L <= 0:
        raise ValueError("SERVER_ETA_L must be > 0")
    if cfg.SERVER_TAU <= 0:
        raise ValueError("SERVER_TAU must be > 0")


def _validate_betas(cfg) -> None:
    """Validate beta parameters for momentum-based optimizers."""
    if not (0 < cfg.BETA_1 < 1):
        raise ValueError("BETA_1 must be in (0, 1)")
    if not (0 < cfg.BETA_2 < 1):
        raise ValueError("BETA_2 must be in (0, 1)")


def _validate_dp(cfg) -> None:
    """Validate differential privacy parameters."""
    if cfg.DP_NOISE_MULTIPLIER < 0:
        raise ValueError("DP_NOISE_MULTIPLIER must be >= 0")
    if cfg.DP_CLIPPING_NORM <= 0:
        raise ValueError("DP_CLIPPING_NORM must be > 0")
    if cfg.DP_NUM_SAMPLED_CLIENTS <= 0:
        raise ValueError("DP_NUM_SAMPLED_CLIENTS must be > 0")

    n = _estimated_train_nodes(cfg)
    if cfg.DP_NUM_SAMPLED_CLIENTS > n:
        raise ValueError(
            "DP_NUM_SAMPLED_CLIENTS cannot exceed expected sampled train nodes per round. "
            f"Got DP_NUM_SAMPLED_CLIENTS={cfg.DP_NUM_SAMPLED_CLIENTS}, estimated n={n}. "
            "Increase FRACTION_TRAIN/NUM_PARTITIONS or reduce DP_NUM_SAMPLED_CLIENTS."
        )

    if cfg.DP_INITIAL_CLIPPING_NORM <= 0:
        raise ValueError("DP_INITIAL_CLIPPING_NORM must be > 0")
    if not (0 <= cfg.DP_TARGET_CLIPPED_QUANTILE <= 1):
        raise ValueError("DP_TARGET_CLIPPED_QUANTILE must be in [0, 1]")
    if cfg.DP_CLIP_NORM_LR <= 0:
        raise ValueError("DP_CLIP_NORM_LR must be > 0")
    if cfg.DP_CLIPPED_COUNT_STDDEV is not None and cfg.DP_CLIPPED_COUNT_STDDEV < 0:
        raise ValueError("DP_CLIPPED_COUNT_STDDEV must be >= 0 when provided")


def _validate_xgb(cfg) -> None:
    """Validate XGBoost-specific parameters."""
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


# ============================================================================
# Strategy-Specific Validators
# ============================================================================
# Each validator function checks parameters specific to that strategy.
# Functions are registered in STRATEGY_VALIDATORS dict for easy lookup.


def _validate_fedavgm(cfg) -> None:
    """Validate FedAvgM (Federated Averaging with Momentum) parameters."""
    if cfg.SERVER_LEARNING_RATE <= 0:
        raise ValueError("SERVER_LEARNING_RATE must be > 0")
    if cfg.SERVER_MOMENTUM < 0:
        raise ValueError("SERVER_MOMENTUM must be >= 0")


def _validate_fedprox(cfg) -> None:
    """Validate FedProx (Proximal Term) parameters."""
    if cfg.PROXIMAL_MU < 0:
        raise ValueError("PROXIMAL_MU must be >= 0")


def _validate_fedadagrad(cfg) -> None:
    """Validate FedAdaGrad parameters."""
    _validate_adaptive_common(cfg)


def _validate_fedadam(cfg) -> None:
    """Validate FedAdam parameters."""
    _validate_adaptive_common(cfg)
    _validate_betas(cfg)


def _validate_fedyogi(cfg) -> None:
    """Validate FedYogi parameters."""
    _validate_adaptive_common(cfg)
    _validate_betas(cfg)


def _validate_fedtrimmedavg(cfg) -> None:
    """Validate FedTrimmedAvg parameters."""
    if not (0 <= cfg.FEDTRIMMEDAVG_BETA < 0.5):
        raise ValueError("FEDTRIMMEDAVG_BETA must be in [0, 0.5)")


def _estimated_train_nodes(cfg) -> int:
    """Estimate per-round sampled train nodes under current sampling settings."""
    return max(2, int(cfg.NUM_PARTITIONS * cfg.FRACTION_TRAIN))


def _validate_bulyan(cfg) -> None:
    """Validate Bulyan (robust aggregation) parameters."""
    if cfg.BULYAN_NUM_MALICIOUS_NODES < 0:
        raise ValueError("BULYAN_NUM_MALICIOUS_NODES must be >= 0")

    n = _estimated_train_nodes(cfg)
    f = cfg.BULYAN_NUM_MALICIOUS_NODES
    if n < 4 * f + 3:
        raise ValueError(
            "Bulyan requires sampled train nodes n >= 4f + 3. "
            f"Current estimate: n={n}, f={f}. "
            "Increase FRACTION_TRAIN/NUM_PARTITIONS or reduce BULYAN_NUM_MALICIOUS_NODES."
        )


def _validate_krum(cfg) -> None:
    """Validate Krum (robust aggregation) parameters."""
    if cfg.KRUM_NUM_MALICIOUS_NODES < 0:
        raise ValueError("KRUM_NUM_MALICIOUS_NODES must be >= 0")

    n = _estimated_train_nodes(cfg)
    f = cfg.KRUM_NUM_MALICIOUS_NODES
    if n < 2 * f + 3:
        raise ValueError(
            "Krum requires sampled train nodes n >= 2f + 3. "
            f"Current estimate: n={n}, f={f}. "
            "Increase FRACTION_TRAIN/NUM_PARTITIONS or reduce KRUM_NUM_MALICIOUS_NODES."
        )


def _validate_multikrum(cfg) -> None:
    """Validate MultiKrum (robust aggregation) parameters."""
    if cfg.MULTIKRUM_NUM_MALICIOUS_NODES < 0:
        raise ValueError("MULTIKRUM_NUM_MALICIOUS_NODES must be >= 0")
    if cfg.MULTIKRUM_NUM_NODES_TO_SELECT <= 0:
        raise ValueError("MULTIKRUM_NUM_NODES_TO_SELECT must be > 0")

    n = _estimated_train_nodes(cfg)
    f = cfg.MULTIKRUM_NUM_MALICIOUS_NODES
    m = cfg.MULTIKRUM_NUM_NODES_TO_SELECT

    if n < 2 * f + 3:
        raise ValueError(
            "MultiKrum requires sampled train nodes n >= 2f + 3. "
            f"Current estimate: n={n}, f={f}. "
            "Increase FRACTION_TRAIN/NUM_PARTITIONS or reduce MULTIKRUM_NUM_MALICIOUS_NODES."
        )

    max_selectable = n - f - 2
    if m > max_selectable:
        raise ValueError(
            "MULTIKRUM_NUM_NODES_TO_SELECT is too large for current (n, f). "
            f"Need m <= n - f - 2, got m={m}, n={n}, f={f}, limit={max_selectable}."
        )


def _validate_qfedavg(cfg) -> None:
    """Validate QFedAvg (q-weighted aggregation) parameters."""
    if not math.isfinite(cfg.QFEDAVG_CLIENT_LEARNING_RATE):
        raise ValueError("QFEDAVG_CLIENT_LEARNING_RATE must be finite")
    if cfg.QFEDAVG_CLIENT_LEARNING_RATE <= 0:
        raise ValueError("QFEDAVG_CLIENT_LEARNING_RATE must be > 0")

    if not math.isfinite(cfg.QFEDAVG_Q):
        raise ValueError("QFEDAVG_Q must be finite")
    if cfg.QFEDAVG_Q < 0:
        raise ValueError("QFEDAVG_Q must be >= 0")


# ============================================================================
# Strategy Validator Registry
# ============================================================================
# Maps each strategy to its validation function.
# To add a new strategy: 1) declare validator function above, 2) add entry here


def _get_strategy_validators(cfg):
    """Build strategy validator registry using cfg for enum references."""
    return {
        cfg.StrategyName.FEDAVG: lambda cfg: None,  # No special validation
        cfg.StrategyName.FEDAVGM: _validate_fedavgm,
        cfg.StrategyName.FEDPROX: _validate_fedprox,
        cfg.StrategyName.FEDADAGRAD: _validate_fedadagrad,
        cfg.StrategyName.FEDADAM: _validate_fedadam,
        cfg.StrategyName.FEDYOGI: _validate_fedyogi,
        cfg.StrategyName.FEDMEDIAN: lambda cfg: None,  # No special validation
        cfg.StrategyName.FEDTRIMMEDAVG: _validate_fedtrimmedavg,
        cfg.StrategyName.BULYAN: _validate_bulyan,
        cfg.StrategyName.KRUM: _validate_krum,
        cfg.StrategyName.MULTIKRUM: _validate_multikrum,
        cfg.StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING: _validate_dp,
        cfg.StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING: _validate_dp,
        cfg.StrategyName.DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING: _validate_dp,
        cfg.StrategyName.DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING: _validate_dp,
        cfg.StrategyName.QFEDAVG: _validate_qfedavg,
        cfg.StrategyName.FEDXGBBAGGING: _validate_xgb,
        cfg.StrategyName.FEDXGBCYCLIC: _validate_xgb,
    }


def _validate_strategy_specific(cfg, strategy) -> None:
    """Validate strategy-specific parameters using registry pattern."""
    validators = _get_strategy_validators(cfg)
    validator = validators.get(strategy)
    
    if validator is None:
        raise ValueError(f"No validator found for strategy: {strategy}")
    
    validator(cfg)


def validate_config_module(cfg) -> None:
    """Validate config module values passed by caller to avoid import cycles."""
    strategy = _resolve_strategy_name(cfg)
    _validate_global(cfg)
    _validate_strategy_specific(cfg, strategy)
