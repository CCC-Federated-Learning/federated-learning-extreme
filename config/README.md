# ⚙️ 配置參數完整指南

本文檔詳細說明所有配置參數、推薦值、以及彼此之間的依賴關係。

---

## 🎯 快速參考表

| 參數 | 類型 | 默認值 | 範圍 | 說明 |
|------|------|--------|------|------|
| `STRATEGY_NAME` | enum | FEDAVG | 見 StrategyName | 聯邦策略選擇 |
| `NUM_ROUNDS` | int | 200 | 50-500 | 聯邦訓練輪數 |
| `LOCAL_EPOCHS` | int | 3 | 1-10 | 客戶端本地訓練輪數 |
| `NUM_PARTITIONS` | int | 10 | 2-100 | 客戶端數量 |
| `BATCH_SIZE` | int | 32 | 8-256 | 本地訓練批次大小 |
| `LEARNING_RATE` | float | 0.01 | 0.0001-0.1 | 客戶端優化器學習率 |
| `DATA_DISTRIBUTION` | enum | IID | IID/LABEL_SKEW/DIRICHLET | 數據分布類型 |
| `FRACTION_TRAIN` | float | 0.9 | 0.7-0.95 | 訓練集比例 |
| `FRACTION_EVALUATE` | float | 0.1 | 0.05-0.3 | 評估集比例 |

---

## 📋 詳細配置說明

### 🚀 核心訓練參數

#### **STRATEGY_NAME** - 聯邦學習策略選擇
```python
from config.types import StrategyName

STRATEGY_NAME = StrategyName.FEDAVG
```

**可選值**（18 種）：
```
FEDAVG, FEDAVGM, FEDPROX,
FEDADAGRAD, FEDADAM, FEDYOGI,
BULYAN, KRUM, MULTIKRUM,
FEDMEDIAN, FEDTRIMMEDAVG,
DP_CLIENT_FIXED, DP_CLIENT_ADAPTIVE, DP_SERVER_FIXED, DP_SERVER_ADAPTIVE,
FEDXGBBAGGING, FEDXGBCYCLIC,
QFEDAVG
```

**推薦**：
- 🌟 **FedAdam** - 通用最穩定
- **FedAvg** - 基線對照
- **FedProx** - 非 IID 數據
- 詳見 [strategies/README.md](../strategies/README.md)

---

#### **NUM_ROUNDS** - 聯邦訓練輪數
```python
NUM_ROUNDS = 200
```

**說明**：
- 每一輪包含：客戶端選擇 → 本地訓練 → 梯度上傳 → 伺服器聚合 → 全局模型更新
- 影響收斂速度和最終精準度

**推薦值**：
| 場景 | 推薦值 | 理由 |
|------|--------|------|
| IID 數據 | 100-150 | 快速收斂 |
| 標籤傾斜 | 200-300 | 需要更多輪緩衝 |
| 差分隱私 | 300-500 | 雜訊需要冗餘輪數 |
| 拜占庭防護 | 300-400 | 保守策略需要更多數據 |
| 快速演示 | 10-50 | 測試用 |

---

#### **LOCAL_EPOCHS** - 本地訓練輪數
```python
LOCAL_EPOCHS = 3
```

**說明**：
- 每個客戶端在本地訓練的 epoch 數
- 越大 → 更新越多，通信次數越少，但本地 drift 越大

**推薦值**：
| 場景 | 推薦值 | 理由 |
|------|--------|------|
| IID 數據 | 1-2 | 最小 drift |
| 非 IID 數據 | 3-5 | FedProx 友好 |
| 高通信成本 | 5-10 | 減少通信輪數 |
| DP 場景 | 1-2 | 隱私保護（每輪加噪聲，輪數少） |

---

#### **NUM_PARTITIONS** - 客戶端數量
```python
NUM_PARTITIONS = 10
```

**說明**：
- 聯邦學習中的「客戶端」數量
- 實際參與訓練的客戶端數 = NUM_PARTITIONS × FRACTION_FIT（通常 100%）

**推薦值**：
| 場景 | 推薦值 | 理由 |
|------|--------|------|
| 學習/演示 | 5-10 | 快速測試 |
| 中等規模 | 10-100 | 典型聯邦場景 |
| 大規模模擬 | 100+ | 真實場景 |

---

### 📊 超參數（Hyperparameters）

#### **BATCH_SIZE** - 本地訓練批次大小
```python
BATCH_SIZE = 32
```

**說明**：
- 每個客戶端本地訓練時的 mini-batch 大小
- 影響本地訓練穩定性和計算速度

**推薦值**：
| 設備 | 推薦值 | 理由 |
|------|--------|------|
| CPU | 8-16 | 記憶體有限 |
| GPU（4GB） | 32 | 平衡點 |
| GPU（8GB） | 64 | 更快 |
| GPU（16GB+） | 128-256 | 最快 |

---

#### **LEARNING_RATE** - 優化器學習率
```python
LEARNING_RATE = 0.01
```

**說明**：
- 客戶端本地優化器的學習率（SGD/Adam 等）
- 控制每步更新幅度

**推薦值**：
| 策略 | 推薦值 | 理由 |
|------|--------|------|
| FedAvg | 0.01-0.05 | 穩定 |
| FedAdam（自適應） | 0.01 | 由 Adam 自動調整 |
| FedProx | 0.005-0.01 | 近端項可能放大 |
| DP 場景 | 0.1-0.5 | 補償雜訊損失 |

**調整規則**：
- 模型發散（NaN/Inf）→ 降低 LR（÷2）
- 收斂太慢 → 提高 LR（×1.5）
- 精準度抖動 → 降低 LR

---

### 📍 數據分布配置

#### **DATA_DISTRIBUTION** - 客戶端數據分布方式
```python
from config.types import DataDistribution

DATA_DISTRIBUTION = DataDistribution.IID
```

**可選值**：

##### 1. **IID** (Independent & Identically Distributed)
```python
DATA_DISTRIBUTION = DataDistribution.IID
```
- **特點**：各客戶端數據完全隨機平均分割
- **數據分布**：每個客戶端都有各類別的樣本（均勻）
- **難度**：⭐ 最簡單
- **適用場景**：
  - ✅ 教學演示
  - ✅ 理論研究（最優下界）
  - ✅ 基線對照
- **推薦超參**：
  ```python
  STRATEGY: FedAvg 或 FedAdam
  NUM_ROUNDS: 100-150
  LOCAL_EPOCHS: 1-2
  ```

---

##### 2. **LABEL_SKEW** (標籤傾斜)
```python
DATA_DISTRIBUTION = DataDistribution.LABEL_SKEW
```
- **特點**：各客戶端只有部分類別的數據
- **極端例子**：客戶端 1 只有「0-4]」，客戶端 2 只有「5-9」
- **難度**：⭐⭐⭐ 較難
- **適用場景**：
  - ✅ 模擬現實（地域性、語言性偏好）
  - ✅ 測試非 IID 抵禦能力
- **推薦超參**：
  ```python
  STRATEGY: FedProx 或 FedAdam
  NUM_ROUNDS: 200-300
  LOCAL_EPOCHS: 3-5
  LEARNING_RATE: 0.005-0.01
  ```

**參數調整**：
```python
# config/__init__.py 中（如有）
LABEL_SKEW_RATIO = 0.5  # 傾斜程度 (0 = IID, 1 = 完全分離)
```

---

##### 3. **DIRICHLET** (Dirichlet 分布)
```python
DATA_DISTRIBUTION = DataDistribution.DIRICHLET
```
- **特點**：基於 Dirichlet(α) 分布的概率採樣
- **自然性**：✅ 最接近現實
- **難度**：⭐⭐⭐⭐ 較難（可調參數多）
- **適用場景**：
  - ✅ 現實模擬
  - ✅ 測試算法魯棒性
- **推薦超參**：
  ```python
  STRATEGY: FedAdam 或 FedProx（αÔ  < 0.5 時）
  NUM_ROUNDS: 200+
  LOCAL_EPOCHS: 3-5
  ```

**參數調整**：
```python
# config/__init__.py 中（如有）
DIRICHLET_ALPHA = 0.5  # α < 1: 高度非均勻
                        # α = 1: 中等非均勻（推薦）
                        # α >> 1: 接近 IID
```

| α 值 | 特性 | 難度 |
|------|------|------|
| 0.1 | 極高度分化 | ⭐⭐⭐⭐⭐ 難 |
| 0.5 | 高度分化 | ⭐⭐⭐⭐ |
| 1.0 | 中等分化（推薦） | ⭐⭐⭐ |
| 10.0 | 低度分化 | ⭐⭐ |
| 100.0 | 接近 IID | ⭐ |

---

#### **FRACTION_TRAIN / FRACTION_EVALUATE** - 訓練/評估集比例
```python
FRACTION_TRAIN = 0.9    # 90% 用於訓練
FRACTION_EVALUATE = 0.1 # 10% 用於評估
```

**說明**：
- 每輪聯邦訓練後，伺服器用評估集計算全局模型精準度
- 與本地資料分割無關（本地也有自己的 train/val）

**推薦值**：
- 標準：0.9 / 0.1（9:1 分割）
- 小數據：0.8 / 0.2（較多評估數據）
- 大數據：0.95 / 0.05（頻繁訓練）

---

### 🔒 策略特定超參數

#### **config/strategy.py** - 各策略的自訂超參

根據不同策略，會有不同的超參需配置：

```python
# ============ 基礎層 ============
# FedAvg: 無額外超參

# FedAvgM: 動量
SERVER_MOMENTUM = 0.9

# FedProx: 近端項
PROXIMAL_MU = 0.01

# ============ 自適應層 ============
# FedAdaGrad: (無需手動配)
# FedAdam / FedYogi: 動量超參
SERVER_MOMENTUM = 0.9
SERVER_MOMENTUM2 = 0.99

# ============ 拜占庭層 ============
# Krum / MultiKrum / Bulyan: (無需手動配)

# ============ 魯棒聚合層 ============
# FedTrimmedAvg: 修剪比例
TRIM_RATIO = 0.1  # 去掉上下各 10%

# ============ 差分隱私層 ============
# 所有 DP_*: 隱私預算
DP_NOISE_MULTIPLIER = 0.5      # 噪聲強度 (重要!)
DP_CLIPPING_NORM = 1.0         # 梯度裁剪範數
DP_CLIPPING_PER_SAMPLE = False # 是否按樣本裁剪

# DP_*_Adaptive 專用: 自適應參數
DP_EPS_TARGET = 1.0            # 目標隱私預算

# ============ XGBoost 層 ============
# FedXgbBagging / FedXgbCyclic:
XGB_ETA = 0.3              # 學習率（Boosting）
XGB_MAX_DEPTH = 6          # 樹深度
XGB_NUM_TREES = 10         # 樹數量

# ============ 公平性層 ============
# QFedAvg: (無需手動配)
```

---

## 🔗 參數依賴關係

```
STRATEGY_NAME
  ├─> 決定是否需要 DP_NOISE_MULTIPLIER
  ├─> 決定是否需要 PROXIMAL_MU
  ├─> 決定是否需要 SERVER_MOMENTUM
  └─> 決定推薦的 NUM_ROUNDS / LOCAL_EPOCHS

DATA_DISTRIBUTION
  ├─> LABEL_SKEW → 推薦 FedProx
  ├─> DIRICHLET → 推薦 FedAdam
  └─> IID → FedAvg 可行

LEARNING_RATE
  ├─> 與 BATCH_SIZE 協調（較大 batch → 更新 LR）
  └─> 與 STRATEGY_NAME 協調
      └─> DP 場景：提高 LR（補償雜訊）
      └─> FedProx：降低 LR（近端項干擾）
```

---

## ✅ 配置檢查清單

在執行 `run.py` 前，按以下清單檢查：

```bash
□ STRATEGY_NAME 已選擇（從 18 種中）
□ NUM_ROUNDS 在合理範圍（50-500）
□ LOCAL_EPOCHS 根據 DATA_DISTRIBUTION 調整
□ BATCH_SIZE 不超過 GPU 記憶體
□ LEARNING_RATE 根據 STRATEGY 設定
□ DATA_DISTRIBUTION 已選擇（IID/LABEL_SKEW/DIRICHLET）
□ 若選 DP：DP_NOISE_MULTIPLIER 已設（推薦 0.1-1.0）
□ 若選 FedProx：PROXIMAL_MU 已設（推薦 0.01-0.1）
□ 若選 XGBoost：XGB_ETA 已設（推薦 0.3）
□ 配置校驗通過：python -c "from app.server import main; from config.validation import validate_config; validate_config()"
```

---

## 🎯 常見配置組合

### 組合 1：快速學習 + IID（默認）
```python
STRATEGY_NAME = StrategyName.FEDAVG
DATA_DISTRIBUTION = DataDistribution.IID
NUM_ROUNDS = 100
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.01
```

### 組合 2：生產環境 + 非 IID
```python
STRATEGY_NAME = StrategyName.FEDADAM
DATA_DISTRIBUTION = DataDistribution.DIRICHLET  # α=0.5
NUM_ROUNDS = 200
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
SERVER_MOMENTUM = 0.9
SERVER_MOMENTUM2 = 0.99
```

### 組合 3：高隱私要求
```python
STRATEGY_NAME = StrategyName.DP_CLIENT_ADAPTIVE
DATA_DISTRIBUTION = DataDistribution.IID
NUM_ROUNDS = 400
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.1        # 提高以補償雜訊
DP_NOISE_MULTIPLIER = 1.0  # 高隱私
DP_CLIPPING_NORM = 1.0
```

### 組合 4：拜占庭防護
```python
STRATEGY_NAME = StrategyName.KRUM
DATA_DISTRIBUTION = DataDistribution.IID
NUM_ROUNDS = 250
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.01
```

### 組合 5：結構化數據 (XGBoost)
```python
STRATEGY_NAME = StrategyName.FEDXGBBAGGING
DATA_DISTRIBUTION = DataDistribution.IID
NUM_ROUNDS = 50
LOCAL_EPOCHS = 1
XGB_ETA = 0.3
XGB_MAX_DEPTH = 6
```

---

## 📝 配置文件模板

複製以下模板到 `config/__init__.py`：

```python
# ============ 核心訓練 ============
from config.types import StrategyName, DataDistribution

STRATEGY_NAME = StrategyName.FEDADAM
NUM_ROUNDS = 200
LOCAL_EPOCHS = 3
NUM_PARTITIONS = 10

# ============ 超參數 ============
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# ============ 數據分布 ============
DATA_DISTRIBUTION = DataDistribution.IID
FRACTION_TRAIN = 0.9
FRACTION_EVALUATE = 0.1

# ============ 策略特定 ============
# FedAdam
SERVER_MOMENTUM = 0.9
SERVER_MOMENTUM2 = 0.99

# FedProx (if using)
# PROXIMAL_MU = 0.01

# DP (if using)
# DP_NOISE_MULTIPLIER = 0.5
# DP_CLIPPING_NORM = 1.0

# XGBoost (if using)
# XGB_ETA = 0.3
# XGB_MAX_DEPTH = 6
```

---

## 🚨 常見錯誤及解決

| 錯誤 | 原因 | 解決 |
|------|------|------|
| NaN / Inf 精準度 | 學習率過高 | 降低 LEARNING_RATE |
| 收斂極慢 | 學習率過低或 NUM_ROUNDS 不足 | 提高 LEARNING_RATE 或 NUM_ROUNDS |
| OOM（記憶體溢出） | BATCH_SIZE 過大 | 降低 BATCH_SIZE |
| DP 場景性能太差 | 噪聲太大 | 降低 DP_NOISE_MULTIPLIER |
| 非 IID 場景發散 | 沒用 FedProx/FedAdam | 切換策略或增大 LOCAL_EPOCHS |

---

## 📚 進階：參數掃描

對多個參數組合進行實驗：

```bash
for lr in 0.001 0.01 0.1; do
  for epochs in 1 3 5; do
    # 修改 config/__init__.py
    sed -i "s/LEARNING_RATE = .*/LEARNING_RATE = $lr/" config/__init__.py
    sed -i "s/LOCAL_EPOCHS = .*/LOCAL_EPOCHS = $epochs/" config/__init__.py
    # 執行
    python run.py
  done
done
```

---

**最後提示**：不確定時，使用默認配置（FedAdam + IID） 即可保證穩定運行！ 🚀
