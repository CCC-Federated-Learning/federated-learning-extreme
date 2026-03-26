from pathlib import Path
import json

input_root = Path(".")
STRATEGY_ORDER = [
    "FedAvg", "FedAvgM", "FedAdagrad", "FedAdam", "FedYogi",
    "FedProx", "FedMedian", "FedTrimmedAvg", "Krum", "MultiKrum", "Bulyan",
    "DifferentialPrivacyServerSideFixedClipping",
    "DifferentialPrivacyServerSideAdaptiveClipping",
    "DifferentialPrivacyClientSideFixedClipping",
    "DifferentialPrivacyClientSideAdaptiveClipping",
    "QFedAvg", "FedXgbBagging", "FedXgbCyclic"
]

batch_dirs = sorted(input_root.iterdir())
for batch_dir in batch_dirs:
    run_root = batch_dir / "PUT-DATA-THERE"
    if not run_root.exists():
        continue

    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        
        metadata_path = run_dir / "metadata.txt"
        metrics_csv = run_dir / "run_metrics.csv"
        metrics_json = run_dir / "run_metrics.json"
        
        if not metadata_path.exists():
            print(f"NO META: {run_dir.name}")
            continue
        
        # Parse metadata
        meta = {}
        with open(metadata_path) as f:
            for line in f:
                if ": " in line:
                    k, v = line.strip().split(": ", 1)
                    meta[k] = v
        
        strategy = meta.get("strategy")
        distribution = meta.get("data_distribution", "").lower()
        
        if strategy not in STRATEGY_ORDER:
            continue
        if distribution not in {"iid", "label"}:
            continue
        
        # Check which format is present
        if metrics_csv.exists():
            print(f"OK CSV:  {distribution:5} {strategy}")
        elif metrics_json.exists():
            print(f"OK JSON: {distribution:5} {strategy}")
        else:
            print(f"NO FILE: {distribution:5} {strategy}")
