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
    FRACTION_TRAIN,
    FRACTION_EVALUATE,
    KRUM_NUM_MALICIOUS_NODES,
    LOCAL_EPOCHS,
    LR,
    MULTIKRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_NODES_TO_SELECT,
    NUM_PARTITIONS,
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


class DataExporter:
    """Export run metrics and metadata files."""

    @staticmethod
    def _strategy_params() -> dict[str, object]:
        """Collect strategy-specific parameters for metadata export."""
        strategy = STRATEGY_NAME

        if strategy == StrategyName.FEDAVG:
            return {}

        if strategy == StrategyName.FEDAVGM:
            return {
                "server_learning_rate": SERVER_LEARNING_RATE,
                "server_momentum": SERVER_MOMENTUM,
            }

        if strategy == StrategyName.FEDPROX:
            return {"proximal_mu": PROXIMAL_MU}

        if strategy == StrategyName.FEDADAGRAD:
            return {
                "server_eta": SERVER_ETA,
                "server_eta_l": SERVER_ETA_L,
                "server_tau": SERVER_TAU,
            }

        if strategy in {StrategyName.FEDADAM, StrategyName.FEDYOGI}:
            return {
                "server_eta": SERVER_ETA,
                "server_eta_l": SERVER_ETA_L,
                "server_tau": SERVER_TAU,
                "beta_1": BETA_1,
                "beta_2": BETA_2,
            }

        if strategy == StrategyName.FEDTRIMMEDAVG:
            return {"fedtrimmedavg_beta": FEDTRIMMEDAVG_BETA}

        if strategy == StrategyName.BULYAN:
            return {"bulyan_num_malicious_nodes": BULYAN_NUM_MALICIOUS_NODES}

        if strategy == StrategyName.KRUM:
            return {"krum_num_malicious_nodes": KRUM_NUM_MALICIOUS_NODES}

        if strategy == StrategyName.MULTIKRUM:
            return {
                "multikrum_num_malicious_nodes": MULTIKRUM_NUM_MALICIOUS_NODES,
                "multikrum_num_nodes_to_select": MULTIKRUM_NUM_NODES_TO_SELECT,
            }

        if strategy in {
            StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING,
            StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING,
            StrategyName.DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING,
            StrategyName.DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING,
        }:
            params: dict[str, object] = {
                "dp_noise_multiplier": DP_NOISE_MULTIPLIER,
                "dp_num_sampled_clients": DP_NUM_SAMPLED_CLIENTS,
                "dp_clipping_norm": DP_CLIPPING_NORM,
            }
            if strategy in {
                StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING,
                StrategyName.DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING,
            }:
                params.update(
                    {
                        "dp_initial_clipping_norm": DP_INITIAL_CLIPPING_NORM,
                        "dp_target_clipped_quantile": DP_TARGET_CLIPPED_QUANTILE,
                        "dp_clip_norm_lr": DP_CLIP_NORM_LR,
                        "dp_clipped_count_stddev": DP_CLIPPED_COUNT_STDDEV,
                    }
                )
            return params

        if strategy == StrategyName.QFEDAVG:
            return {
                "qfedavg_client_learning_rate": QFEDAVG_CLIENT_LEARNING_RATE,
                "qfedavg_q": QFEDAVG_Q,
            }

        if strategy in {StrategyName.FEDXGBBAGGING, StrategyName.FEDXGBCYCLIC}:
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
    def save_json(experiment_dir: Path, records: list[dict[str, float | int]]) -> Path:
        output_path = experiment_dir / "run_metrics.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, indent=2)
        return output_path

    @staticmethod
    def save_csv(experiment_dir: Path, records: list[dict[str, float | int]]) -> Path:
        output_path = experiment_dir / "run_metrics.csv"
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["round", "accuracy", "loss"])
            writer.writeheader()
            writer.writerows(records)
        return output_path

    @staticmethod
    def save_metadata(
        experiment_dir: Path,
        run_id: str,
        model_name: str,
        elapsed_seconds: float,
    ) -> Path:
        metadata_path = experiment_dir / "metadata.txt"
        common_fields = [
            ("run_id", run_id),
            ("title", f"{DATASET_NAME}-{DATA_DISTRIBUTION}-{STRATEGY_NAME}"),
            ("model_name", model_name),
            ("dataset_name", DATASET_NAME),
            ("data_distribution", DATA_DISTRIBUTION),
            ("strategy", STRATEGY_NAME),
            ("num_rounds", NUM_ROUNDS),
            ("local_epochs", LOCAL_EPOCHS),
            ("batch_size", BATCH_SIZE),
            ("learning_rate", LR),
            ("fraction_train", FRACTION_TRAIN),
            ("fraction_evaluate", FRACTION_EVALUATE),
            ("dirichlet_alpha", DIRICHLET_ALPHA),
            ("data_seed", DATA_SEED),
            ("num_partitions", NUM_PARTITIONS),
            ("elapsed_seconds", f"{elapsed_seconds:.2f}"),
        ]

        lines = ["# Common"]
        lines.extend(f"{key} = {value}" for key, value in common_fields)

        strategy_params = DataExporter._strategy_params()
        lines.append("")
        lines.append("# StrategySpecific")
        if strategy_params:
            lines.extend(
                f"{key} = {value}"
                for key, value in sorted(strategy_params.items(), key=lambda item: item[0])
            )
        else:
            lines.append("none = true")

        content = "\n".join(lines) + "\n"
        metadata_path.write_text(content, encoding="utf-8")
        return metadata_path
