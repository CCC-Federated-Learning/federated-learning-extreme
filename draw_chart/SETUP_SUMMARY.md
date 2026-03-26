# 🎯 Draw Chart Pipeline - Setup Summary

## ✅ Configuration Complete

Your three-stage processing pipeline is now fully configured and ready to use!

---

## 📊 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 01: Input & Generation                                │
│ ├─ User Input Dir: PUT-DATA-THERE/                        │
│ ├─ Output Dir: generate_charts/                             │
│ ├─ Main Script: generator.py (existing template)            │
│ └─ Transfer Script: move_to_stage02.py ✨ NEW               │
└─────────────────────────────────────────────────────────────┘
            ↓ (python move_to_stage02.py)
┌─────────────────────────────────────────────────────────────┐
│ STAGE 02: Processing & Analysis                             │
│ ├─ Input Dir: PUT-DATA-THERE/ (auto-filled from stage01)    │
│ ├─ Output Dir: generate_charts/                             │
│ ├─ Main Script: generator.py (existing template)            │
│ ├─ Validation: clearner.py (existing template)              │
│ └─ Transfer Script: move_to_stage03.py ✨ NEW               │
└─────────────────────────────────────────────────────────────┘
            ↓ (python move_to_stage03.py)
┌─────────────────────────────────────────────────────────────┐
│ STAGE 03: Final Archive                                     │
│ ├─ Input Dir: PUT-DATA-THERE/ (auto-filled from stage02)    │
│ ├─ Output Dir: generate_charts/                             │
│ └─ Status: Read-Only Archive ✓                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📝 Files Created

### New Scripts

#### 1. `stage01/move_to_stage02.py` ✨
- **Purpose**: Transfer stage01 outputs to stage02
- **Function**: 
  - Extracts experiment metadata from `PUT-DATA-THERE/`
  - Creates appropriately named directory in stage02
  - Copies `PUT-DATA-THERE/` → `stage02/{exp_dir}/PUT-DATA-THERE/`
  - Copies `generate_charts/` → `stage02/{exp_dir}/generate_charts/`
  - Creates SUMMARY.txt with metadata

#### 2. `stage02/move_to_stage03.py` ✨
- **Purpose**: Transfer stage02 outputs to finally_result
- **Function**:
  - Extracts experiment metadata from `PUT-DATA-THERE/`
  - Creates appropriately named directory in finally_result
  - Copies `PUT-DATA-THERE/` → `finally_result/{exp_dir}/PUT-DATA-THERE/`
  - Copies `generate_charts/` → `finally_result/{exp_dir}/generate_charts/`
  - Creates SUMMARY.txt with metadata

### New Documentation

#### 1. `draw_chart/README.md` ✨
- Master documentation for the entire pipeline
- Quick start guide for all three stages
- Integration information
- Use case examples

#### 2. `stage01/STAGE01_README.md` ✨
- Detailed guide for Stage 01
- Input data format and requirements
- Configuration options for generator.py
- Troubleshooting

#### 3. `stage02/STAGE02_README.md` ✨
- Detailed guide for Stage 02
- Data processing and analysis steps
- Cleaning and validation procedures
- Configuration and performance tuning

#### 4. `finally_result/FINALLY_RESULT_README.md` ✨
- Guide for the final archive stage
- How to access and organize results
- Reporting and exporting tips
- FAQ

### Utility Scripts

#### 1. `quick_start.sh` ✨
- Bash script for quick status checking
- Useful on macOS/Linux

#### 2. `quick_start.ps1` ✨
- PowerShell script for Windows
- Shows current pipeline status with emojis
- Command reference

---

## 📁 Directory Structure Created

```
draw_chart/
├── README.md ✨                    # Master documentation
├── quick_start.sh ✨               # Bash quick reference
├── quick_start.ps1 ✨              # PowerShell quick reference
│
├── stage01/
│   ├── PUT-DATA-THERE/ ✨         # ← User places raw data HERE
│   ├── generate_charts/            # Generator.py output
│   ├── move_to_stage02.py ✨       # New transfer script
│   ├── STAGE01_README.md ✨        # Detailed documentation
│   └── ... (existing templates from res-drag-set-to-there)
│
├── stage02/
│   ├── PUT-DATA-THERE/ ✨          # Auto-filled from stage01
│   ├── generate_charts/            # Generator.py output  
│   ├── move_to_stage03.py ✨       # New transfer script
│   ├── STAGE02_README.md ✨        # Detailed documentation
│   └── ... (existing templates from res-happy-ending)
│
└── stage03/
    ├── PUT-DATA-THERE/ ✨          # Auto-filled from stage02
    ├── STAGE03_README.md ✨        # Archive documentation
    └── generate_charts/            # Final visualization output
```

---

## 🚀 How to Use

### Quick Start (5 minutes)

```bash
# Step 1: Prepare input data
cp -r /path/to/raw_data stage01/PUT-DATA-THERE/

# Step 2: Run Stage 01
cd stage01
python generator.py
python move_to_stage02.py

# Step 3: Run Stage 02
cd ../stage02
python generator.py
python move_to_stage03.py

# Step 4: View results
cd ../stage03
ls -la PUT-DATA-THERE/
find . -name "*.png"
```

### Detailed Workflow

1. **Stage 01 Input**
   ```bash
   # Place your experiment data
   mkdir stage01/PUT-DATA-THERE/experiment_001
   cp metadata.txt run_metrics.csv stage01/PUT-DATA-THERE/experiment_001/
   ```

2. **Stage 01 Processing**
   ```bash
   cd stage01
   python generator.py          # Generates charts locally
   python move_to_stage02.py   # Sends to stage02
   ```

3. **Stage 02 Processing**
   ```bash
   cd ../stage02
   # Data automatically arrives in PUT-DATA-THERE/
   python generator.py          # Secondary analysis
   python clearner.py          # Optional: validate data
   python move_to_stage03.py   # Archive to finally_result
   ```

4. **Stage 03 Results**
   ```bash
   cd ../finally_result
   # Browse results: 
   ls */generate_charts/
   ```

---

## 🔧 Configuration

### Stage 01 Configuration (in generator.py)
- `STRATEGY_GROUPS`: Define visualization groups
- `COLORS`: Color palette for charts
- `DP_SHORT_NAMES`: Friendly names for strategies

### Stage 02 Configuration (in generator.py)
- `MIN_ACCURACY_THRESHOLD`: Filter results
- `OUTLIER_REMOVAL`: Statistical filtering
- `NUM_WORKERS`: Parallel processing threads

### Environment Variables (Optional)
```bash
export DRAW_CHART_LOG_LEVEL=INFO
export DRAW_CHART_WORKERS=4
export DRAW_CHART_CACHE=true
```

---

## 📊 Data Requirements

### Input Format (put-data/)
Each experiment must have:

```
put-data/
└── experiment_name/
    ├── metadata.txt
    │   data_distribution=IID          (required)
    │   num_rounds=200                 (required)
    │   local_epochs=3                 (required)
    │   run_id=20260324-2043-57        (required)
    │   num_clients=10
    │   strategy_name=FedAvg
    │   ...
    ├── run_metrics.csv
    │   round,accuracy,loss,communication_cost
    │   0,0.123,2.451,512
    │   ...
    └── other_optional_files.csv
```

### Output Format (generate_charts/)
- `accuracy_comparison.png`
- `loss_curves.png`
- `convergence_analysis.png`
- `*.csv` with statistics
- `SUMMARY.txt` with metadata

---

## ✨ Key Features

### Automatic Features
- ✅ Metadata extraction from experiment files
- ✅ Automatic directory naming convention
- ✅ SUMMARY.txt generation at each stage
- ✅ Overwrite protection with user prompts
- ✅ Validation of required metadata fields

### Data Flow
- ✅ Forward-only pipeline (stage01→02→03)
- ✅ Non-destructive copying (original data preserved)
- ✅ Flexible: pause/resume between stages
- ✅ Resumable: can re-run transfer scripts anytime

### Integration
- ✅ Compatible with existing `generator.py` scripts
- ✅ Uses standardized `metadata.txt` format
- ✅ Works with both IID and Label data distributions
- ✅ Supports all 18 FL strategies

---

## 📤 Exporting Results

### For Presentations
```bash
# Copy all final charts
cp finally_result/*/generate_charts/*.png ~/presentations/

# Or specific strategy comparison
cp finally_result/*/generate_charts/performance_ranking.png ~/reports/
```

### For Publications
```bash
# Export numerical data
cat finally_result/*/generate_charts/detailed_metrics.csv | head -20

# Get statistics in JSON (if available)
cat finally_result/*/generate_charts/statistics.json
```

### For Archiving
```bash
# Compress entire pipeline output
tar -czf stage03_results_$(date +%Y%m%d).tar.gz stage03/
```

---

## 🔄 Related Directories

### In Project Root
- `raw_data/` - Original experiment outputs (read-only backup)
- `experiment_results/` - Alternative final storage option

### Stage Templates (Reference)
- `res-drag-set-to-there----first-station/` - Provides generator.py template
- `res-happy-ending--------second-station/` - Provides analysis templates

---

## ⚠️ Important Notes

### Data Preservation
- ✓ Original data in `PUT-DATA-THERE/` is NOT deleted
- ✓ All outputs are COPIED (not moved) to next stage
- ✓ Stage 03 is the final archive - treat as read-only

### Naming Convention
Directories in stages follow this pattern:
```
{DataDistribution}-Round-{NumRounds}-Epoch-{LocalEpochs}-{RunID}

Examples:
IID-Round-200-Epoch-3-20260324-2043-57
Label-Round-100-Epoch-2-20260325-1234-56
```

### File Structure Expectations
- Each stage directory is self-contained
- `PUT-DATA-THERE/` is the input directory
- `generate_charts` are output directories
- Scripts read from input, write to output

---

## 🐛 Troubleshooting

### Stage 02 doesn't receive data from Stage 01
```bash
# Re-run the transfer script
cd stage01
python move_to_stage02.py
```

### "put-data directory not found" error
```bash
# Create it manually
mkdir stage01/put-data

# Add your data
cp -r /experiments/* stage01/put-data/
```

### "Could not find valid metadata.txt" error
```bash
# Verify metadata.txt is in each experiment directory
# Required fields:
data_distribution=IID
num_rounds=200
local_epochs=3
run_id=20260324-2043-57
```

### Clean restart
```bash
# Safely clear outputs (preserving input data)
rm -rf stage01/generate_charts/*
rm -rf stage02/PUT-DATA-THERE/*
rm -rf stage02/generate_charts/*
rm -rf stage03/PUT-DATA-THERE/*
rm -rf stage03/generate_charts/*
```

---

## 📚 Documentation Map

| Document | Location | Purpose |
|----------|----------|---------|
| **Master README** | `draw_chart/README.md` | Pipeline overview & quick start |
| **Stage 01 Guide** | `stage01/STAGE01_README.md` | Input & generation details |
| **Stage 02 Guide** | `stage02/STAGE02_README.md` | Processing & analysis details |
| **Stage 03 Guide** | `stage03/STAGE03_README.md` | Archive & export details |
| **This File** | `draw_chart/SETUP_SUMMARY.md` | Setup verification & reference |

---

## ✅ Verification Checklist

- [x] `draw_chart/` directory created
- [x] `stage01/`, `stage02/`, `stage03/` created
- [x] `stage01/PUT-DATA-THERE/` created (user input location)
- [x] `stage02/PUT-DATA-THERE/` created (stage01 output)
- [x] `stage03/PUT-DATA-THERE/` created (stage02 output)
- [x] `move_to_stage02.py` script created
- [x] `move_to_stage03.py` script created
- [x] Documentation created (4 detailed README files)
- [x] Quick start scripts created (bash & PowerShell)
- [x] Integration validated with existing templates
- [x] Directory naming convention standardized

---

## 🎉 You're All Set!

Your three-stage pipeline is ready to process federated learning experiments!

**Next steps:**
1. Read `README.md` in this directory for an overview
2. Place your experiment data in `stage01/PUT-DATA-THERE/`
3. Run `python generator.py` in stage01
4. Run `python move_to_stage02.py` to transfer
5. Repeat steps 3-4 for stage02 & stage03
6. Access final results in `stage03/`

**Questions?**
- Refer to `STAGE0X_README.md` for stage-specific details
- Check `../raw_data/` for original experiment data
- Use templates from `res-drag-set-to-there` and `res-happy-ending` for custom generators

Happy analyzing! 📊✨

