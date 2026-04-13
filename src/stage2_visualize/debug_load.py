from pathlib import Path
import pandas as pd
import json

# Test loading the two problematic files
base = Path("data/label-Round-200-Epoch-3-20260322-2312-59/data")

strategies = [
    "20260325-0434-48-DifferentialPrivacyClientSideAdaptiveClipping",
    "20260325-0504-08-DifferentialPrivacyServerSideAdaptiveClipping",
]

for strat_dir in strategies:
    csv_path = base / strat_dir / "metrics.csv"
    json_path = base / strat_dir / "metrics.json"
    
    print(f"\n{strat_dir.split('-')[-1]}:")
    
    # Try CSV
    try:
        df_csv = pd.read_csv(csv_path)
        print(f"  CSV loaded: {df_csv.shape}")
        print(f"  Columns: {list(df_csv.columns)}")
        print(f"  First row: {df_csv.iloc[0].to_dict() if len(df_csv) > 0 else 'empty'}")
    except Exception as e:
        print(f"  CSV error: {e}")
    
    # Try JSON
    try:
        with open(json_path) as f:
            json_data = json.load(f)
            if isinstance(json_data, dict) and "rounds" in json_data:
                json_data = json_data["rounds"]
            df_json = pd.DataFrame(json_data)
            print(f"  JSON loaded: {df_json.shape}")
            print(f"  Columns: {list(df_json.columns)}")
    except Exception as e:
        print(f"  JSON error: {e}")
