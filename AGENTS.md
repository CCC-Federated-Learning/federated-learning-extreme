# Federated Learning Extreme — Agent Reference

聯邦學習研究平台，基於 **Flower 1.29.0**，實作並比較 **18 種 FL 策略**。

---

## 快速開始

```bash
pip install -r requirements.txt       # 安裝依賴

# 編輯 src/config/__init__.py 設定策略與超參
python run.py                         # 執行單一策略
python batch_run.py                   # 批次執行多個策略
```

執行結果自動存入 `results/runs/{timestamp}-{strategy}/`。

---

## 目錄結構

```
federated-learning-extreme/
│
├── run.py                    # 單一策略執行入口
├── batch_run.py              # 批次執行多個策略
├── plot_distribution.py      # 繪製客戶端資料分布圖
├── check_gpu.py              # 確認 GPU 可用性
├── requirements.txt
│
├── src/
│   ├── config/
│   │   ├── __init__.py       # ★ 主設定檔（策略、超參）
│   │   ├── strategy.py       # 各策略專屬超參
│   │   ├── types.py          # StrategyName / DataDistribution / DatasetName
│   │   └── validation.py     # 執行前設定驗證
│   │
│   ├── app/
│   │   ├── server.py         # ServerApp：初始化、聚合、全局評估
│   │   ├── client.py         # ClientApp（PyTorch）：本地訓練、DP 裁剪
│   │   ├── client_xgb.py     # ClientApp（XGBoost）
│   │   ├── model.py          # CNN 模型 + 資料分割 + train_model/eval_model
│   │   ├── model_xgb.py      # XGBoost 資料載入
│   │   └── data_cache.py     # MNIST 資料集與 partition 快取
│   │
│   ├── strategies/
│   │   ├── factory.py        # 策略工廠（build_strategy()）
│   │   ├── fedavg.py ... qfedavg.py   # 18 個策略建構函式
│   │
│   ├── record/
│   │   ├── recorder.py       # Recorder：記錄每輪指標、呼叫 finalize
│   │   ├── exporter.py       # Exporter：輸出 metrics.json/csv、metadata.txt
│   │   └── chart.py          # plot_accuracy_chart()
│   │
│   ├── stage1_collect/       # Pipeline 第一站：收集與驗證結果
│   │   ├── generate.py       # 讀取 data/，產生比較圖表至 charts/
│   │   ├── check.py          # 確認 18 個策略都已執行
│   │   ├── export.py         # 將 data/ + charts/ 搬到 stage2_visualize/
│   │   └── cleaner.py        # 清空 data/ 和 charts/
│   │
│   └── stage2_visualize/     # Pipeline 第二站：IID vs Label 跨策略視覺化
│       ├── generate.py       # 讀取 data/，產生 IID vs Label 對比圖
│       ├── export.py         # 搬到 results/final/{dir_name}/
│       ├── debug_load.py     # 偵錯：測試資料載入
│       └── cleaner.py        # 清空 data/ 和 charts/
│
└── results/
    ├── runs/                 # 每次 run.py 的輸出（時間戳資料夾）
    ├── data/                 # MNIST 資料集（首次執行自動下載）
    ├── charts/               # plot_distribution.py 的分布圖
    ├── models/               # 訓練完成的模型（final_model.pt / .json）
    └── final/                # stage2 export.py 的最終輸出
```

---

## 設定檔：`src/config/__init__.py`

唯一需要手動修改的檔案。

```python
# ─── 在這裡改策略 ────────────────────────────────
STRATEGY_NAME = StrategyName.FEDAVG
DATA_DISTRIBUTION = DataDistribution.IID
DATASET_NAME = DatasetName.MNIST

# ─── 訓練設定 ─────────────────────────────────────
NUM_ROUNDS   = 200
LOCAL_EPOCHS = 3
BATCH_SIZE   = 64
LR           = 0.001
NUM_CLIENTS  = 10

FRACTION_TRAIN    = 1.0
FRACTION_EVALUATE = 1.0

# ─── 資料分布設定 ──────────────────────────────────
DIRICHLET_ALPHA = 0.5   # 越小越 non-IID
DATA_SEED       = 499

# ─── 執行環境設定 ──────────────────────────────────
CLIENT_NUM_CPUS              = 2.0
CLIENT_NUM_GPUS_IF_AVAILABLE = 0.1
```

策略專屬超參（`DP_NOISE_MULTIPLIER`、`PROXIMAL_MU`、`SERVER_ETA` 等）在 `src/config/strategy.py`。

### 資料分布說明

| 分布 | 說明 | 適用場景 |
|------|------|----------|
| `IID` | 完全隨機均勻分割 | 基準對照 |
| `LABEL` | 每個客戶端只有部分標籤 | 極端 non-IID |
| `DIRICHLET` | Dirichlet(α) 概率採樣 | 現實模擬，α↓ 越 non-IID |

---

## 18 個聯邦學習策略

### 基礎

| 策略 | 特點 | 建議用途 |
|------|------|----------|
| `FedAvg` | 加權平均，無額外超參 | 基準 |
| `FedAvgM` | 伺服器端動量 (`SERVER_MOMENTUM`) | 加速收斂 |
| `FedProx` | 近端正則化 (`PROXIMAL_MU`)，抑制 local drift | non-IID 數據 |

### 自適應優化（伺服器端）

| 策略 | 特點 |
|------|------|
| `FedAdagrad` | AdaGrad；參數：`SERVER_ETA`, `SERVER_ETA_L`, `SERVER_TAU` |
| `FedAdam` | Adam；額外需要 `BETA_1`, `BETA_2`；**通用首選** |
| `FedYogi` | Yogi（改良 Adam）；對梯度異常值更穩健 |

### Byzantine 防護

| 策略 | 容忍惡意節點 | 特點 |
|------|-------------|------|
| `Krum` | ~20% | 選距離最近的客戶端 |
| `MultiKrum` | ~20% | 多輪 Krum，更穩定 |
| `Bulyan` | ~25% | 最強防護，計算量最高 |

### 魯棒聚合

| 策略 | 特點 |
|------|------|
| `FedMedian` | 座標中位數，抑制離群值 |
| `FedTrimmedAvg` | 修剪均值，參數 `FEDTRIMMEDAVG_BETA`（建議 0.1–0.2） |

### 差分隱私（4 種）

裁剪位置 × 裁剪方式：

| 策略 | 裁剪位置 | 裁剪方式 |
|------|----------|----------|
| `DifferentialPrivacyClientSideFixedClipping` | 客戶端 | 固定範數 |
| `DifferentialPrivacyClientSideAdaptiveClipping` | 客戶端 | 自適應範數 |
| `DifferentialPrivacyServerSideFixedClipping` | 伺服器 | 固定範數 |
| `DifferentialPrivacyServerSideAdaptiveClipping` | 伺服器 | 自適應範數 |

關鍵超參：`DP_NOISE_MULTIPLIER`、`DP_CLIPPING_NORM`、`DP_NUM_SAMPLED_CLIENTS`。
- 客戶端 DP：裁剪在 `src/app/client.py` 執行（`compute_adaptive_clip_model_update`）
- 伺服器 DP：由 Flower 策略包裝器處理

### XGBoost

| 策略 | 特點 |
|------|------|
| `FedXgbBagging` | 每輪各客戶端獨立訓練樹 |
| `FedXgbCyclic` | 接續全局模型繼續疊加樹 |

MNIST 圖片會被壓平為 784 維表格特徵。超參在 `src/config/strategy.py` 的 `XGB_*` 區段。

### 公平性

| 策略 | 特點 |
|------|------|
| `QFedAvg` | 提高高損失客戶端的聚合權重，參數 `QFEDAVG_Q` |

### 策略選擇速查

| 情境 | 建議策略 |
|------|----------|
| IID 基準 | `FedAvg` |
| non-IID，通用 | `FedAdam` |
| non-IID，local drift 嚴重 | `FedProx` |
| 隱私要求高 | `DPClientAdaptiveClipping` |
| 有惡意客戶端 | `Krum` / `Bulyan` |
| 結構化表格資料 | `FedXgbBagging` |

---

## 執行流程

### 單一策略

1. 編輯 `src/config/__init__.py`，設定 `STRATEGY_NAME`、`NUM_ROUNDS` 等
2. `python run.py`
3. 結果存入 `results/runs/{timestamp}-{strategy}/`

每個結果資料夾包含：
```
results/runs/20260413-2230-31-FedAvg/
├── metrics.json        # 每輪 accuracy + loss（JSON）
├── metrics.csv         # 每輪 accuracy + loss（CSV）
├── metadata.txt        # 設定快照（策略、超參、耗時）
├── accuracy.png        # accuracy vs round 圖
└── distribution.png    # 客戶端資料分布圖
```

`metadata.txt` 格式：
```ini
# Common
strategy = FedAvg
distribution = IID
num_rounds = 200
local_epochs = 3
lr = 0.001
num_clients = 10
elapsed_seconds = 1234.56

# Strategy
none = true        # 或策略專屬超參
```

### 批次執行

編輯 `batch_run.py` 的 `STRATEGIES_TO_RUN` 清單，然後：

```bash
python batch_run.py
```

腳本會依序 patch `src/config/__init__.py`、呼叫 `run.py`，完成後還原設定檔。

---

## 結果分析 Pipeline

### 第一站 `src/stage1_collect/`

**職責**：從 `results/runs/` 收集結果，產生統計圖表。

```
data/     ← 把 results/runs/ 中的目錄複製到這裡
charts/   ← generate.py 輸出圖表至此
```

```bash
cd src/stage1_collect/
python generate.py    # 讀取 data/，產生 charts/
python check.py       # 確認 18 個策略都有結果
python export.py      # 將 data/ + charts/ 搬到 stage2_visualize/{dir}/
python cleaner.py     # 清空 data/ 和 charts/（重新開始用）
```

### 第二站 `src/stage2_visualize/`

**職責**：比較 IID vs Label Skew 兩種分布，產生跨策略對比圖。

```
data/     ← 從 stage1 export 接收批次資料夾
charts/   ← generate.py 輸出對比圖
```

```bash
cd src/stage2_visualize/
python generate.py    # 產生 IID vs Label 對比圖（36 條曲線視覺化）
python export.py      # 搬到 results/final/{distribution}-Round-{n}-Epoch-{e}-{run_id}/
python cleaner.py     # 清空暫存
```

### 最終輸出 `results/final/`

```
results/final/IID-label-Round-200-Epoch-3-20260325-xxxx/
├── data/            # 原始各策略指標
├── charts/          # 對比圖表
└── SUMMARY.txt
```

---

## 模組參考

### `src/app/model.py` — PyTorch 模型與資料

**`Net`**：2 層 CNN（Conv→Pool→Conv→Pool→FC→FC），約 35K 參數，28×28 灰階輸入。

**`load_client_data(client_id, num_clients, batch_size, distribution, ...)`**
→ `(trainloader, testloader)`

**`load_test_dataloader()`** → 全量測試集 DataLoader

**`train_model(net, trainloader, epochs, lr, device, proximal_mu, global_params)`**
→ `float`（平均 training loss）

**`eval_model(net, dataloader, device)`**
→ `(loss, accuracy)`

**`_build_partitions(targets, num_clients, distribution, dirichlet_alpha, seed)`**
→ `list[list[int]]`（每個客戶端的樣本索引）

### `src/app/server.py` — ServerApp

- 初始化全局模型（PyTorch）或空陣列（XGBoost）
- 每輪呼叫 `_eval_global()` 或 `_eval_global_xgb()` 評估
- 訓練完成後儲存模型至 `results/models/`
- 使用 `Recorder` 記錄每輪指標

### `src/app/client.py` — ClientApp（PyTorch）

- 從 `msg.content["arrays"]` 接收全局模型
- 呼叫 `train_model()` 執行本地訓練
- 若 config 含 `KEY_CLIPPING_NORM`，執行 DP 梯度裁剪
- 回傳更新後的 `ArrayRecord` + `MetricRecord`

### `src/record/recorder.py` — Recorder

```python
recorder = Recorder()
recorder.start()                              # 建立 results/runs/{run_id}/ 目錄
recorder.record_round(round_idx, acc, loss)   # 每輪記錄
recorder.finalize()                           # 輸出 JSON/CSV/chart/metadata
```

### `src/strategies/factory.py` — 策略工廠

```python
from strategies.factory import build_strategy
strategy = build_strategy()   # 根據 config.STRATEGY_NAME 建構對應策略
```

---

## 套件依賴

```
flwr[simulation]   1.29.0  — Flower FL 框架
torch              2.11.0  — PyTorch
torchvision        0.26.0  — MNIST 資料集
numpy              2.4.4   — 數值運算
matplotlib         3.10.8  — 圖表輸出
xgboost                    — XGBoost 策略（可選）
ipywidgets                 — Jupyter 進度顯示
```

---

## 常見問題

**Q: 如何切換策略？**
編輯 `src/config/__init__.py` 的 `STRATEGY_NAME`，執行 `python run.py`。

**Q: 如何調整 non-IID 程度？**
`DATA_DISTRIBUTION = DataDistribution.DIRICHLET`，調低 `DIRICHLET_ALPHA`（0.1 = 極端 non-IID）。

**Q: DP 策略精準度很差？**
降低 `DP_NOISE_MULTIPLIER`（試 0.1–0.5），或增加 `NUM_ROUNDS`。

**Q: Byzantine 策略設定限制？**
Krum/MultiKrum 要求 `n >= 2f+3`，Bulyan 要求 `n >= 4f+3`（n = 訓練客戶端數，f = `*_NUM_MALICIOUS_NODES`）。`validate_config()` 會在執行前檢查。

**Q: XGBoost 策略報錯？**
確認已安裝 `xgboost`：`pip install xgboost`。

**Q: 結果存在哪裡？**
`results/runs/{timestamp}-{strategy}/`，含 `metrics.csv`、`metadata.txt`、`accuracy.png`。

**Q: 如何讀取結果做分析？**
```python
import pandas as pd
df = pd.read_csv("results/runs/20260413-2230-31-FedAvg/metrics.csv")
print(df["accuracy"].iloc[-1])   # 最終精準度
```
