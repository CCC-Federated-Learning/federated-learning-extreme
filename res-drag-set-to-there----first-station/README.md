# 🚩 First Station: 收集、清理、驗證結果

**res-drag-set-to-there----first-station** 是三站點流程中的**第一關口**，負責從原始實驗結果中收集、驗證和清理數據。

---

## 🎯 職責一覽

| 步驟 | 檔案 | 什麼做 |
|------|------|------|
| 1️⃣ **生成統計** | `generator.py` | 讀取 `res/history/` 的結果，計算統計 |
| 2️⃣ **驗證完整性** | `check_strategies.py` | 確保 18 個策略都執行成功 |
| 3️⃣ **清理失敗執行** | `clearner.py` | 刪除不完整或錯誤的執行紀錄 |
| 4️⃣ **整理搬運** | `move_outer.py` | 最終組織並移至 Second Station |

---

## 📊 輸入與輸出

### 輸入
```
res/history/
├── 20260324-2043-57/        # 時間戳資料夾（每次執行一個）
│   ├── run_metrics.csv
│   ├── metadata.txt
│   ├── accuracy_chart.png
│   └── distribution_chart.png
├── 20260324-2232-26/
├── 20260325-0100-00/
└── ...
```

### 輸出
```
PUT-DATA-THERE/
├── IID-Round-20-Epoch-1-20260324-2043-57/     # 動態命名
│   ├── run_metrics.csv
│   └── metadata.txt
├── label-Round-20-Epoch-1-20260324-2232-26/
└── ...
```

---

## 🔧 使用步驟

### 步驟 1：執行所有策略

在根目錄執行：
```bash
python run_selected_strategies.py
```

這會依序執行 18 個策略，結果保存到 `res/history/{timestamp}/`

---

### 步驟 2：轉到 First Station

```bash
cd res-drag-set-to-there----first-station
```

---

### 步驟 3：運行完整流程

```bash
# ⚠️ 大坑：PUT-DATA-THERE 有舊東西時會詢問是否覆蓋，改用自動覆蓋版本

python move_outer.py
```

**單一腳本完成所有步驟**（推薦）：
- ✅ 自動讀取 `res/history/`
- ✅ 驗證佈總體完整性
- ✅ 清理失敗執行
- ✅ 搬運到 `PUT-DATA-THERE/`

---

## 📖 詳細說明

### **generator.py** - 統計生成器

```python
def generate_statistics():
    """掃描 res/history/ 並生成統計"""
    for timestamp_dir in sorted(Path("../res/history").glob("????????-????-??")):
        # 讀取 metadata.txt
        metadata = read_metadata(timestamp_dir / "metadata.txt")
        
        # 讀取 CSV 指標
        df = pd.read_csv(timestamp_dir / "run_metrics.csv")
        
        # 計算統計
        stats = {
            "strategy": metadata["strategy_name"],
            "final_accuracy": df["accuracy"].iloc[-1],
            "final_loss": df["loss"].iloc[-1],
            "convergence_round": find_convergence(df),  # 首次達 90% 的輪數
            "total_time": metadata.get("total_training_time_seconds", "N/A"),
        }
        
        print(f"{timestamp_dir.name}: {stats}")
```

**輸出**：
```
Result Statistics:
────────────────────────
20260324-2043-57 (IID):
  Strategy: FedAdam
  Final Accuracy: 0.956
  Convergence (90%): Round 45
  Time: 2345.6s

20260324-2232-26 (label):
  Strategy: FedProx
  Final Accuracy: 0.934
  Convergence (90%): Round 78
  Time: 3021.2s
```

---

### **check_strategies.py** - 驗證器

```python
def check_all_strategies_completed():
    """檢查 18 個策略是否都執行過"""
    from config.types import StrategyName
    
    completed_strategies = set()
    
    # 掃描 res/history/ 並收集已執行的策略
    for result_dir in Path("../res/history").glob("*"):
        metadata_file = result_dir / "metadata.txt"
        if metadata_file.exists():
            strategy = read_metadata_field(metadata_file, "strategy_name")
            completed_strategies.add(strategy)
    
    # 對比 18 個預期策略
    expected = {s.name for s in StrategyName}
    missing = expected - completed_strategies
    
    if missing:
        print(f"❌ Missing strategies: {missing}")
        return False
    else:
        print(f"✅ All 18 strategies completed!")
        return True
```

---

### **clearner.py** - 清潔器

```python
def clean_incomplete_runs():
    """移除不完整的執行（缺少 CSV 或 metadata）"""
    incomplete = []
    
    for result_dir in Path("../res/history").glob("*"):
        required_files = [
            result_dir / "run_metrics.csv",
            result_dir / "metadata.txt"
        ]
        
        if not all(f.exists() for f in required_files):
            incomplete.append(result_dir)
            print(f"Removing incomplete: {result_dir.name}")
            shutil.rmtree(result_dir)
    
    if not incomplete:
        print("✅ No incomplete runs found")
```

---

### **move_outer.py** - 搬運整理器

**最新版本**（改寫過的）：

```python
def aggregate_results():
    """最終搬運：所有結果 → PUT-DATA-THERE/"""
    
    # 1. 掃描 res/history/
    history_dir = Path("../res/history")
    
    # 2. 驗證完整性
    if not check_all_strategies_completed():
        print("⚠️ Not all strategies completed, continuing anyway...")
    
    # 3. 清理失敗執行
    clean_incomplete_runs()
    
    # 4. 搬運到 PUT-DATA-THERE/
    put_data_dir = Path("PUT-DATA-THERE")
    put_data_dir.mkdir(exist_ok=True)
    
    for result_dir in sorted(history_dir.glob("*")):
        # 從 metadata 提取配置信息
        config = extract_experiment_config(result_dir)
        
        # 動態生成資料夾名稱
        # 格式：{distribution}-Round-{rounds}-Epoch-{epochs}-{run_id}
        folder_name = create_directory_name(config)
        
        # 複製檔案（使用 robocopy 處理長路徑）
        subprocess.run([
            'robocopy', 
            str(result_dir), 
            str(put_data_dir / folder_name),
            '/E', '/MT:4'
        ])
        
        print(f"✅ Moved {result_dir.name} → {folder_name}")
    
    print(f"\n✅ All results organized in PUT-DATA-THERE/")
```

**關鍵參數提取**：

從 `metadata.txt` 中提取：
- `data_distribution`: IID、label、dirichlet 等
- `num_rounds`: 200, 300 等
- `local_epochs`: 1, 3, 5 等
- `run_id`: 時間戳 (20260324-2043-57)

生成的資料夾名稱示例：
```
IID-Round-200-Epoch-3-20260324-2043-57
label-Round-200-Epoch-3-20260324-2232-26
dirichlet-Round-100-Epoch-5-20260325-0100-00
```

---

## 📋 完整執行檢查清單

```bash
# 0️⃣ 前置準備
□ 根目錄確實有 run_selected_strategies.py
□ res/history/ 有最新的時間戳資料夾

# 1️⃣ 執行所有策略（可選，若已執行可跳過）
python ../run_selected_strategies.py
# 大約需要 24-48 小時…

# 2️⃣ 來到 first-station
cd res-drag-set-to-there----first-station

# 3️⃣ 驗證並搬運（一鍵完成）
python move_outer.py

# 4️⃣ 檢查輸出
ls PUT-DATA-THERE/
# 應該看到多個 {distribution}-Round-... 資料夾

# 5️⃣ 查看統計
python -c "
import pandas as pd
from pathlib import Path
for d in Path('PUT-DATA-THERE').glob('*'):
    csv_file = d / 'run_metrics.csv'
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f'{d.name}: Acc={df.accuracy.iloc[-1]:.4f}')
"
```

---

## 🚨 常見問題

### Q1: 執行卡住了怎麼辦？
A: 
- `move_outer.py` 可能詢問是否覆蓋（改寫後應自動覆蓋，無需手動）
- 如果卡住，按 `Ctrl+C` 中止，手動刪除 `PUT-DATA-THERE/` 再試

### Q2: 為什麼 check_strategies.py 說有策略缺失？
A:
- 執行 `run_selected_strategies.py` 時中途斷開
- 某些策略因超參不合法被跳過
- 檢查 `res/history/` 下最新的資料夾

### Q3: robocp 有錯誤怎麼辦？
A:
- Windows 路徑太長（超過 260 字符）
- 確保 Windows 有啟用長路徑支持
- 或手動進行複製/剪貼操作

### Q4: 可以只搬運某些策略嗎？
A:
- 編輯 `move_outer.py`，在「遍歷歷史資料夾」的迴圈中加入過濾條件
- 例如：只搬運 IID 的結果

---

## 🔗 下一步

✅ First Station 完成 → 進入 **Second Station**

```bash
cd ../res-happy-ending--------second-station
python generator.py        # 生成對比圖表
python move_outer.py       # 最終整理到 RESULT_Finally_fan_tasty/
```

詳見 [res-happy-ending--------second-station/README.md](../res-happy-ending--------second-station/README.md)

---

**快速查詢**：[根目錄 README.md](../README.md) | [三站點流程](../README.md#-三站點資料流程)
