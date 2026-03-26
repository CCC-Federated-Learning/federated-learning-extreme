# 📊 Draw Chart - Stage 02: Data Processing & Analysis

## Overview
**Stage 02** receives processed data from Stage 01 and performs secondary analysis and filtering.

### Workflow
1. **Input**: Receives data from Stage 01 via `move_to_stage02.py`
2. **Processing**: Filter, clean, and analyze experimental results
3. **Output**: Generated insights and processed charts
4. **Transfer**: `move_to_stage03.py` transfers final aggregated results to Finally_Result

---

## 📁 Directory Structure

```
stage02/
├── PUT-DATA-THERE/              # ← Data from Stage 01 (auto-created)
│   └── {timestamp-experiments}/
│       ├── metadata.txt
│       └── run_metrics.csv
├── generate_charts/             # Stage 02 analysis outputs
│   └── *.png, *.csv
├── generator.py                 # Analysis and visualization script
├── move_to_stage03.py          # Transfer script to Stage 03
├── clearner.py                 # Optional: clean incomplete runs
└── README.md                    # This file
```

---

## 🚀 Quick Start

### Step 1: Wait for Data from Stage 01
```bash
# Data will be automatically placed here:
PUT-DATA-THERE/{experiment_directories}/
```

### Step 2: Generate Analysis
```bash
python generator.py
```
This will:
- Read all experiments from `PUT-DATA-THERE/`
- Perform statistical analysis
- Create comparison charts
- Save outputs to `generate_charts/`

### Step 3: (Optional) Clean Suspicious Results
```bash
python clearner.py
```
This will:
- Identify incomplete experiments
- Check for data integrity
- Remove runs with missing files (optional)

### Step 4: Move to Stage 03
```bash
python move_to_stage03.py
```
This will:
- Extract experiment configuration
- Create directory in `../stage03/`
- Copy all processed data and charts
- Create SUMMARY.txt

---

## 📋 Processing Steps

### Automatic Data Validation
```
✓ Check metadata.txt integrity
✓ Validate CSV format
✓ Verify required metrics columns
✓ Extract experiment parameters
```

### Analysis Options

**Basic Analysis:**
```bash
python generator.py --mode basic
```

**Detailed Comparison:**
```bash
python generator.py --mode detailed
```

**Strategy Ranking:**
```bash
python generator.py --mode ranking
```

---

## 🔍 Data Quality Checks

### clearner.py Validation

```bash
python clearner.py --check
```

Checks for:
- Missing metadata.txt files
- Incomplete CSV data
- Corrupted metrics
- Missing required columns

Remove problematic runs:
```bash
python clearner.py --cleanup
```

---

## 📊 Expected Outputs

After `python generator.py`:

```
generate_charts/
├── strategy_comparison.png
├── accuracy_vs_rounds.png
├── convergence_speed.png
├── robustness_analysis.png
├── performance_ranking.png
├── detailed_metrics.csv
└── statistical_summary.json
```

---

## 🔗 Integration with Stages

### Receives from:
- **Stage 01**: Via `move_to_stage02.py`
  - Gets `PUT-DATA-THERE/{experiment_dirs}/`
  - Gets `generate_charts/` from stage01

### Sends to:
- **Stage 03**: Via `move_to_stage03.py`
  - Copies `PUT-DATA-THERE/` → `stage03/{exp_dir}/PUT-DATA-THERE/`
  - Copies `generate_charts/` → `stage03/{exp_dir}/generate_charts/`

---

## ⚙️ Configuration

### Filtering Options (in generator.py)

```python
MIN_ACCURACY_THRESHOLD = 0.80
MIN_ROUNDS_REQUIRED = 50
REQUIRED_STRATEGIES = ['FedAvg', 'FedProx', 'FedAdam']
OUTLIER_REMOVAL = True
```

### Comparison Groups

```python
STRATEGY_GROUPS = {
    'Basic': ['FedAvg', 'FedAvgM', 'FedProx'],
    'Adaptive': ['FedAdagrad', 'FedAdam', 'FedYogi'],
    'Robust': ['Bulyan', 'Krum', 'MultiKrum'],
}
```

---

## 📝 Output Formats

### CSV Summary
```csv
Strategy,Min_Accuracy,Max_Accuracy,Avg_Convergence_Round,Total_Time_Sec
FedAvg,0.92,0.96,45,2345.6
FedProx,0.91,0.95,56,2512.3
...
```

### JSON Metadata
```json
{
  "experiment_id": "IID-Round-200-Epoch-3-20260324-2043-57",
  "strategies_analyzed": 18,
  "total_runs": 54,
  "data_quality": "valid",
  "processing_timestamp": "2026-03-26T14:30:45Z"
}
```

---

## 🔄 Reprocessing Data

To reprocess without Stage 01 interference:

```bash
# 1. Keep PUT-DATA-THERE but clear outputs
rm -r generate_charts/*

# 2. Regenerate with new parameters
python generator.py --config custom_config.ini

# 3. Move to stage03
python move_to_stage03.py
```

---

## ❌ Common Issues

### "PUT-DATA-THERE directory not found"
```
Solution: Run move_to_stage02.py from Stage 01 first
```

### Analysis fails on certain metrics
```
Solution: Run python clearner.py --check
         Then python clearner.py --cleanup
```

### Memory error with large datasets
```
Solution: Process subsets manually or increase RAM
         python generator.py --chunk-size 10
```

---

## 📌 Notes

- **Processing Time**: Depends on data size (typically 5-30 min)
- **Output Size**: ~100-500 MB depending on number of experiments
- **Parallelization**: Set `NUM_WORKERS=4` in generator.py for faster processing
- **Caching**: `generate_charts/.cache/` stores intermediate results

