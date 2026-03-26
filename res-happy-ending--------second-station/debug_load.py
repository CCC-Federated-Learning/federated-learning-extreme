import pandas as pd
import re
from pathlib import Path

input_root = Path('PUT-DATA-THERE')
STRATEGY_ORDER = [
    'FedAvg', 'FedAvgM', 'FedAdagrad', 'FedAdam', 'FedProx', 'FedYogi', 'Bulyan', 'Krum', 'MultiKrum',
    'FedMedian', 'FedTrimmedAvg',
    'DifferentialPrivacyClientSideAdaptiveClipping',
    'DifferentialPrivacyClientSideFixedClipping',
    'DifferentialPrivacyServerSideAdaptiveClipping',
    'DifferentialPrivacyServerSideFixedClipping',
    'FedXgbBagging', 'FedXgbCyclic', 'QFedAvg',
]

data = {'iid': {}, 'label': {}}
batch_dirs = [d for d in sorted(input_root.iterdir()) if d.is_dir()]

print("=== Data Loading Debug ===\n")

for batch_dir in batch_dirs:
    run_root = batch_dir / 'PUT-DATA-THERE'
    if not run_root.exists():
        print(f"Run root not found: {run_root}")
        continue
    
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        
        metadata_path = run_dir / 'metadata.txt'
        metrics_path = run_dir / 'run_metrics.csv'
        
        if not metadata_path.exists():
            print(f"NO METADATA: {run_dir.name}")
            continue
        if not metrics_path.exists():
            print(f"NO CSV: {run_dir.name}")
            continue
        
        # Parse metadata
        with open(metadata_path, 'r') as f:
            content = f.read()
            match = re.search(r'strategy = (\w+)', content)
            strategy = match.group(1) if match else None
            dist_match = re.search(r'data_distribution = (\w+)', content)
            distribution = dist_match.group(1).lower() if dist_match else None
        
        if strategy not in STRATEGY_ORDER:
            print(f'SKIPPED (strategy): {run_dir.name:50} - {strategy}')
            continue
        if distribution not in {'iid', 'label'}:
            print(f'SKIPPED (distribution): {run_dir.name:50} - {distribution}')
            continue
        
        print(f'[LOADED] {strategy:35} / {distribution:5} / {run_dir.name}')

print("\n=== Summary ===")
print(f"Total checked directories: {len(list(input_root.rglob('metadata.txt')))}")
