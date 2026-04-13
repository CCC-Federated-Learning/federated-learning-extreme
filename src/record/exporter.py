import csv
import json
from pathlib import Path

from config import (
    BATCH_SIZE,
    BETA_1,
    BETA_2,
    BULYAN_NUM_MALICIOUS_NODES,
    DATASET_NAME,
    DATA_DISTRIBUTION,
    DATA_SEED,
    DIRICHLET_ALPHA,
    DP_CLIPPED_COUNT_STDDEV,
    DP_CLIP_NORM_LR,
    DP_CLIPPING_NORM,
    DP_INITIAL_CLIPPING_NORM,
    DP_NOISE_MULTIPLIER,
    DP_NUM_SAMPLED_CLIENTS,
    DP_TARGET_CLIPPED_QUANTILE,
    FEDTRIMMEDAVG_BETA,
    FRACTION_EVALUATE,
    FRACTION_TRAIN,
    KRUM_NUM_MALICIOUS_NODES,
    LOCAL_EPOCHS,
    LR,
    MULTIKRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_NODES_TO_SELECT,
    NUM_CLIENTS,
    NUM_ROUNDS,
    PROXIMAL_MU,
    QFEDAVG_CLIENT_LEARNING_RATE,
    QFEDAVG_Q,
    SERVER_ETA,
    SERVER_ETA_L,
    SERVER_LEARNING_RATE,
    SERVER_MOMENTUM,
    SERVER_TAU,
    STRATEGY_NAME,
    StrategyName,
    XGB_COLSAMPLE_BYTREE,
    XGB_ETA,
    XGB_MAX_DEPTH,
    XGB_MIN_CHILD_WEIGHT,
    XGB_NUM_CLASS,
    XGB_NUM_LOCAL_ROUND,
    XGB_OBJECTIVE,
    XGB_REG_LAMBDA,
    XGB_SUBSAMPLE,
)


class Exporter:
    """Write metrics and metadata files for a completed run."""

    @staticmethod
    def _strategy_params() -> dict[str, object]:
        s = STRATEGY_NAME
        if s == StrategyName.FEDAVG:
            return {}
        if s == StrategyName.FEDAVGM:
            return {"server_learning_rate": SERVER_LEARNING_RATE, "server_momentum": SERVER_MOMENTUM}
        if s == StrategyName.FEDPROX:
            return {"proximal_mu": PROXIMAL_MU}
        if s == StrategyName.FEDADAGRAD:
            return {"server_eta": SERVER_ETA, "server_eta_l": SERVER_ETA_L, "server_tau": SERVER_TAU}
        if s in {StrategyName.FEDADAM, StrategyName.FEDYOGI}:
            return {
                "server_eta": SERVER_ETA, "server_eta_l": SERVER_ETA_L,
                "server_tau": SERVER_TAU, "beta_1": BETA_1, "beta_2": BETA_2,
            }
        if s == StrategyName.FEDMEDIAN:
            return {}
        if s == StrategyName.FEDTRIMMEDAVG:
            return {"fedtrimmedavg_beta": FEDTRIMMEDAVG_BETA}
        if s == StrategyName.BULYAN:
            return {"bulyan_num_malicious_nodes": BULYAN_NUM_MALICIOUS_NODES}
        if s == StrategyName.KRUM:
            return {"krum_num_malicious_nodes": KRUM_NUM_MALICIOUS_NODES}
        if s == StrategyName.MULTIKRUM:
            return {
                "multikrum_num_malicious_nodes": MULTIKRUM_NUM_MALICIOUS_NODES,
                "multikrum_num_nodes_to_select": MULTIKRUM_NUM_NODES_TO_SELECT,
            }
        if s in {
            StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING,
            StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING,
            StrategyName.DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING,
            StrategyName.DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING,
        }:
            params: dict[str, object] = {
                "dp_noise_multiplier": DP_NOISE_MULTIPLIER,
                "dp_clipping_norm": DP_CLIPPING_NORM,
                "dp_num_sampled_clients": DP_NUM_SAMPLED_CLIENTS,
            }
            if s in {
                StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING,
                StrategyName.DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING,
            }:
                params.update({
                    "dp_initial_clipping_norm": DP_INITIAL_CLIPPING_NORM,
                    "dp_target_clipped_quantile": DP_TARGET_CLIPPED_QUANTILE,
                    "dp_clip_norm_lr": DP_CLIP_NORM_LR,
                    "dp_clipped_count_stddev": DP_CLIPPED_COUNT_STDDEV,
                })
            return params
        if s == StrategyName.QFEDAVG:
            return {
                "qfedavg_client_lr": QFEDAVG_CLIENT_LEARNING_RATE,
                "qfedavg_q": QFEDAVG_Q,
            }
        if s in {StrategyName.FEDXGBBAGGING, StrategyName.FEDXGBCYCLIC}:
            return {
                "xgb_objective": XGB_OBJECTIVE,
                "xgb_num_class": XGB_NUM_CLASS,
                "xgb_num_local_round": XGB_NUM_LOCAL_ROUND,
                "xgb_eta": XGB_ETA,
                "xgb_max_depth": XGB_MAX_DEPTH,
                "xgb_subsample": XGB_SUBSAMPLE,
                "xgb_colsample_bytree": XGB_COLSAMPLE_BYTREE,
                "xgb_min_child_weight": XGB_MIN_CHILD_WEIGHT,
                "xgb_reg_lambda": XGB_REG_LAMBDA,
            }
        return {}

    @staticmethod
    def save_json(run_dir: Path, rounds: list[dict[str, float | int]]) -> Path:
        path = run_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(rounds, f, indent=2)
        return path

    @staticmethod
    def save_csv(run_dir: Path, rounds: list[dict[str, float | int]]) -> Path:
        path = run_dir / "metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "accuracy", "loss"])
            writer.writeheader()
            writer.writerows(rounds)
        return path

    @staticmethod
    def save_metadata(
        run_dir: Path,
        run_id: str,
        model_name: str,
        elapsed_seconds: float,
    ) -> Path:
        common = [
            ("run_id",            run_id),
            ("title",             f"{DATASET_NAME}-{DATA_DISTRIBUTION}-{STRATEGY_NAME}"),
            ("model_name",        model_name),
            ("dataset",           DATASET_NAME),
            ("distribution",      DATA_DISTRIBUTION),
            ("strategy",          STRATEGY_NAME),
            ("num_rounds",        NUM_ROUNDS),
            ("local_epochs",      LOCAL_EPOCHS),
            ("batch_size",        BATCH_SIZE),
            ("lr",                LR),
            ("fraction_train",    FRACTION_TRAIN),
            ("fraction_evaluate", FRACTION_EVALUATE),
            ("dirichlet_alpha",   DIRICHLET_ALPHA),
            ("data_seed",         DATA_SEED),
            ("num_clients",       NUM_CLIENTS),
            ("elapsed_seconds",   f"{elapsed_seconds:.2f}"),
        ]

        lines = ["# Common"]
        lines.extend(f"{k} = {v}" for k, v in common)

        strategy_params = Exporter._strategy_params()
        lines.append("")
        lines.append("# Strategy")
        if strategy_params:
            lines.extend(f"{k} = {v}" for k, v in sorted(strategy_params.items()))
        else:
            lines.append("none = true")

        path = run_dir / "metadata.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
