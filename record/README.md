# 📊 結果記錄與導出 (record/)

本模組負責收集、整理、導出聯邦學習訓練過程中的所有指標和結果。

---

## 📁 檔案結構

| 檔案 | 職責 |
|------|------|
| **recorder.py** | 主控制器 - 協調整個記錄流程 |
| **data_recorder.py** | 指標收集 - 存儲每輪的精準度、損失等 |
| **data_exporter.py** | 數據導出 - 生成 JSON、CSV、metadata.txt |
| **draw_acc_chart.py** | 圖表繪製 - 生成精準度曲線 PNG |

---

## 🔄 記錄流程

```
ServerApp 訓練循環
    │
    ├─ 每一輪結束
    │   └─ metrics = {"accuracy": 0.95, "loss": 0.123}
    │
    ▼
recorder.record_round(round_num, metrics)
    │
    ├─ data_recorder.add_metrics()
    │   └─ 存儲到 list：[round_1_metrics, round_2_metrics, ...]
    │
    ├─ (可選) 即時畫圖
    │   └─ draw_acc_chart.plot_partial()
    │
    └─ 進到下一輪...
    
[訓練完成]
    │
    ▼
recorder.finalize()
    │
    ├─ data_exporter.to_json()
    │   └─ metrics.json
    │
    ├─ data_exporter.to_csv()
    │   └─ run_metrics.csv
    │
    ├─ data_exporter.to_metadata()
    │   └─ metadata.txt (超參、耗時、策略名等)
    │
    └─ draw_acc_chart.plot_final()
        └─ accuracy_chart.png
```

---

## 🔍 詳細模組說明

### **recorder.py** - 主記錄器

#### 職責
- 協調其他記錄模組
- 管理輸出目錄
- 處理記錄生命週期

#### 關鍵類

##### `MetricsRecorder` 類
```python
class MetricsRecorder:
    def __init__(self, output_dir: str, config: dict):
        """初始化記錄器
        
        Args:
            output_dir: 輸出目錄（如 res/history/20260325-XXXX-XX/）
            config: 配置字典（含 STRATEGY_NAME, NUM_ROUNDS, 等）
        """
        self.output_dir = Path(output_dir)
        self.config = config
        
        # 初始化子模組
        self.data_recorder = DataRecorder()
        self.exporter = DataExporter(output_dir)
        self.plotter = AccuracyPlotter()
    
    def record_round(self, round_num: int, metrics: dict):
        """記錄單一輪的指標
        
        Args:
            round_num: 輪數 (0-indexed)
            metrics: {"accuracy": 0.95, "loss": 0.123}
        """
        # 1. 存儲指標
        self.data_recorder.add_metrics(round_num, metrics)
        
        # 2. (可選) 即時輸出
        print(f"[Round {round_num+1}/{self.config['NUM_ROUNDS']}]")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Loss: {metrics['loss']:.4f}")
    
    def finalize(self, final_metrics: dict):
        """訓練完成後的最終處理
        
        Args:
            final_metrics: 最終指標（通常是最後一輪）
        """
        # 1. 導出數據
        self.exporter.to_json(self.data_recorder.get_all_metrics())
        self.exporter.to_csv(self.data_recorder.get_all_metrics())
        self.exporter.to_metadata(
            config=self.config,
            final_metrics=final_metrics,
            num_rounds=self.data_recorder.get_num_rounds()
        )
        
        # 2. 生成圖表
        self.plotter.plot(
            metrics=self.data_recorder.get_all_metrics(),
            output_path=self.output_dir / "accuracy_chart.png"
        )
        
        print(f"\n✅ Results saved to: {self.output_dir}")
```

---

### **data_recorder.py** - 指標收集

#### 職責
- 存儲每一輪的指標
- 提供查詢和統計功能

#### 關鍵類

##### `DataRecorder` 類
```python
class DataRecorder:
    def __init__(self):
        self.metrics = []  # 列表：[{round: 0, accuracy: ..., loss: ...}, ...]
    
    def add_metrics(self, round_num: int, metrics: dict):
        """添加一輪的指標"""
        record = {"round": round_num, **metrics}
        self.metrics.append(record)
    
    def get_all_metrics(self) -> list[dict]:
        """返回所有指標"""
        return self.metrics
    
    def get_round_metrics(self, round_num: int) -> dict:
        """查詢特定輪的指標"""
        return self.metrics[round_num]
    
    def get_num_rounds(self) -> int:
        """返回總輪數"""
        return len(self.metrics)

# 示例數據結構
metrics = [
    {"round": 0, "accuracy": 0.098, "loss": 2.301},
    {"round": 1, "accuracy": 0.156, "loss": 2.251},
    {"round": 2, "accuracy": 0.198, "loss": 2.187},
    # ... 200 輪
]
```

---

### **data_exporter.py** - 數據導出

#### 職責
- 導出指標為多種格式
- 生成 metadata.txt 記錄超參

#### 導出格式

##### 1. **JSON 格式** (`metrics.json`)
```json
{
  "rounds": [
    {
      "round": 0,
      "accuracy": 0.098,
      "loss": 2.301
    },
    {
      "round": 1,
      "accuracy": 0.156,
      "loss": 2.251
    }
  ],
  "summary": {
    "final_accuracy": 0.956,
    "final_loss": 0.142,
    "total_rounds": 200
  }
}
```

**用途**：程序化讀取、分析

---

##### 2. **CSV 格式** (`run_metrics.csv`)
```csv
round,accuracy,loss
0,0.098,2.301
1,0.156,2.251
2,0.198,2.187
...
199,0.956,0.142
```

**用途**：Excel 分析、圖表製作、數據對比

**關鍵欄位說明**：
- **round**: 聯邦訓練輪數 (0-indexed)
- **accuracy**: 全局模型在評估集上的精準度 (0-1)
- **loss**: 全局模型在評估集上的損失 (≥0)

---

##### 3. **Metadata 格式** (`metadata.txt`)
```
# Federated Learning Experiment Metadata
# Generated: 2026-03-25 14:30:00

[STRATEGY]
strategy_name=FedAdam
file=app/server.py

[TRAINING]
num_rounds=200
local_epochs=3
num_clients=10
num_sampled_clients=10

[DATA]
data_distribution=IID
training_set_fraction=0.9
evaluation_set_fraction=0.1
batch_size=32

[OPTIMIZATION]
learning_rate=0.01
server_momentum=0.9
server_momentum2=0.99

[RESULTS]
final_accuracy=0.956
final_loss=0.142
total_training_time_seconds=2345.6

[DEVICE]
device=cpu
num_gpus=0
```

**用途**：實驗復現、參數查詢、自動化對比

---

#### 關鍵函數

##### `to_json(metrics: list[dict], output_path: Path)`
```python
def to_json(self, metrics, output_path):
    """導出為 JSON"""
    data = {
        "rounds": metrics,
        "summary": {
            "final_accuracy": metrics[-1]["accuracy"],
            "final_loss": metrics[-1]["loss"],
            "total_rounds": len(metrics)
        }
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
```

##### `to_csv(metrics: list[dict], output_path: Path)`
```python
def to_csv(self, metrics, output_path):
    """導出為 CSV"""
    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
```

##### `to_metadata(config: dict, final_metrics: dict, num_rounds: int, output_path: Path)`
```python
def to_metadata(self, config, final_metrics, num_rounds, output_path):
    """生成 metadata.txt"""
    content = f"""# Federated Learning Experiment Metadata
# Generated: {datetime.now().isoformat()}

[STRATEGY]
strategy_name={config['STRATEGY_NAME']}

[TRAINING]
num_rounds={num_rounds}
local_epochs={config['LOCAL_EPOCHS']}
num_clients={config['NUM_PARTITIONS']}

[DATA]
data_distribution={config['DATA_DISTRIBUTION']}
batch_size={config['BATCH_SIZE']}

[OPTIMIZATION]
learning_rate={config['LEARNING_RATE']}

[RESULTS]
final_accuracy={final_metrics['accuracy']:.6f}
final_loss={final_metrics['loss']:.6f}
run_id={self.config.get('RUN_ID', 'unknown')}
"""
    with open(output_path, "w") as f:
        f.write(content)
```

---

### **draw_acc_chart.py** - 圖表繪製

#### 職責
- 繪製精準度 vs 輪數的曲線圖
- 支援即時更新（訓練中）和最終生成

#### 關鍵函數

##### `plot(metrics: list[dict], output_path: Path, title: str = "")`
```python
def plot(metrics, output_path, title=""):
    """繪製精準度曲線"""
    rounds = [m["round"] for m in metrics]
    accuracies = [m["accuracy"] for m in metrics]
    losses = [m["loss"] for m in metrics]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 精準度曲線
    ax1.plot(rounds, accuracies, linewidth=2, color='blue')
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Global Model Accuracy - {title}")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # 損失曲線
    ax2.plot(rounds, losses, linewidth=2, color='red')
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"Global Model Loss - {title}")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Chart saved to {output_path}")
```

**輸出範例**：
```
accuracy_chart.png (寬 1200px × 高 400px)
├─ 左圖：精準度曲線（藍色）
└─ 右圖：損失曲線（紅色）
```

---

## 📁 輸出目錄結構

執行 `python run.py` 後，結果會保存到：

```
res/history/{timestamp}/
├── run_metrics.csv              # 核心指標表
├── metadata.txt                 # 超參紀錄
├── accuracy_chart.png           # 精準度圖表
├── distribution_chart.png       # 數據分布（由 draw_distribution.py 生成）
└── final_model.pt               # 最終模型權重（可選）

範例時間戳：20260325-143000 (2026年3月25日 14:30:00)
```

---

## 📊 結果分析範例

### 讀取 CSV 進行分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 1. 讀取結果
df = pd.read_csv("res/history/20260325-143000/run_metrics.csv")

# 2. 查看摘要
print(df.describe())
#          round   accuracy       loss
# count  200.000000  200.000000  200.000000
# mean    99.500000    0.750000    0.587000
# std     57.734800    0.276000    0.701000
# min      0.000000    0.098000    0.142000
# 25%     49.750000    0.456000    0.289000
# 50%     99.500000    0.892000    0.456000
# 75%    149.250000    0.934000    0.234000
# max    199.000000    0.956000    2.301000

# 3. 最終精準度
final_accuracy = df["accuracy"].iloc[-1]
print(f"Final Accuracy: {final_accuracy:.4f}")

# 4. 收斂速度（首次達到 90% 精準度的輪數）
convergence_round = df[df["accuracy"] >= 0.90].index[0]
print(f"Convergence Round (90% acc): {convergence_round}")

# 5. 繪製自訂圖表
fig, ax = plt.subplots()
ax.plot(df["round"], df["accuracy"], label="Accuracy", marker='o', markersize=2)
ax.axhline(y=0.9, color='r', linestyle='--', label="90% Threshold")
ax.set_xlabel("Round")
ax.set_ylabel("Accuracy")
ax.legend()
plt.savefig("custom_chart.png")
```

---

### 讀取 metadata.txt 進行對比

```python
import configparser

# 讀取超參
config = configparser.ConfigParser()
config.read("res/history/20260325-143000/metadata.txt")

print(f"Strategy: {config['STRATEGY']['strategy_name']}")
print(f"Rounds: {config['TRAINING']['num_rounds']}")
print(f"Data Distribution: {config['DATA']['data_distribution']}")
print(f"Final Accuracy: {config['RESULTS']['final_accuracy']}")
```

---

## 🔄 多實驗對比

### 自動化對比腳本

```python
import pandas as pd
from pathlib import Path

# 收集多個實驗結果
results = []

for run_dir in Path("res/history").glob("*"):
    csv_file = run_dir / "run_metrics.csv"
    metadata_file = run_dir / "metadata.txt"
    
    if csv_file.exists() and metadata_file.exists():
        df = pd.read_csv(csv_file)
        
        config = configparser.ConfigParser()
        config.read(metadata_file)
        
        results.append({
            "run_id": run_dir.name,
            "strategy": config['STRATEGY']['strategy_name'],
            "final_accuracy": df["accuracy"].iloc[-1],
            "final_loss": df["loss"].iloc[-1],
            "convergence_round": df[df["accuracy"] >= 0.90].index[0] if any(df["accuracy"] >= 0.90) else -1,
        })

# 彙總為表格
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

# 輸出最佳策略
best_strategy = summary_df.loc[summary_df["final_accuracy"].idxmax()]
print(f"\n🏆 Best Strategy: {best_strategy['strategy']}")
print(f"   Final Accuracy: {best_strategy['final_accuracy']:.4f}")
```

---

## 🚨 常見問題

### Q1: 為什麼精準度一直不變？
A: 檢查：
- 模型是否正確初始化
- 學習率是否太小
- 數據是否正確加載

### Q2: 如何導出中間結果？
A: `DataExporter.to_csv()` 可在訓練中任何時候調用

### Q3: 如何自定義導出格式？
A: 編輯 `data_exporter.py`，添加新的導出函數（如 XML、HDF5 等）

---

**更多信息**：見 [record/ 原始碼](.) 或根目錄 [README.md](../README.md)
