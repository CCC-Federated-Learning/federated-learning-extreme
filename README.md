# 🚀 Federated Learning Extreme

## 📖 項目概述

一個**comprehensive 聯邦學習研究平台**，基於 [Flower 1.25.0](https://flower.ai) 框架，實現並對比 **18 種聯邦學習策略**，覆蓋基礎平均、拜占庭防護、差分隱私、自適應優化、XGBoost 訓練等多個維度。

**核心用途**：系統化評估不同聯邦學習算法在分布式訓練環境下的性能、隱私、魯棒性權衡。

---

## ⚡ 快速開始（5 分鐘）

### 1. 環境安裝
```bash
pip install -r requirements.txt
```

### 2. 執行單個策略（以 FedAvg 為例）
```bash
# 編輯 config/__init__.py，設置：
# STRATEGY_NAME = StrategyName.FEDAVG
# NUM_ROUNDS = 200
# LOCAL_EPOCHS = 3
# DATA_DISTRIBUTION = DataDistribution.IID

python run.py
```

### 3. 批次執行全部 18 策略
```bash
python run_selected_strategies.py
```

執行結果會保存到 `res/history/{timestamp}/` 目錄下。

---

## 🏗️ 項目架構

### 目錄結構速覽
```
federated-learning-extreme/
├── README.md                    # ← 你在這裡
├── config/                      # 全局配置管理
│   ├── __init__.py              # ⭐ 主控開關（策略、超參選擇）
│   ├── strategy.py              # 策略超參（DP、FedProx、XGBoost 等）
│   ├── types.py                 # 枚舉定義（18 策略、3 種分布）
│   └── validation.py            # 配置驗證
│
├── app/                         # 聯邦學習應用程式
│   ├── server.py                # ServerApp：模型初始化、聚合、評估
│   ├── client.py                # ClientApp：本地訓練、DP 應用
│   ├── task.py                  # PyTorch 模型 + 數據加載（MNIST）
│   ├── task_xgb.py              # XGBoost 任務
│   ├── client_xgb.py            # XGBoost 客戶端
│   └── data_cache.py            # 數據集緩存
│
├── strategies/                  # 18 個聯邦演算法實現
│   ├── fedavg.py, fedavgm.py, fedprox.py    # 基礎
│   ├── fedadagrad.py, fedadam.py, fedyogi.py # 自適應
│   ├── bulyan.py, krum.py, multikrum.py     # Byzantine 防護
│   ├── fedmedian.py, fedtrimmedavg.py       # 魯棒聚合
│   ├── dp_client_*.py, dp_server_*.py       # 差分隱私 (4 種)
│   ├── fedxgbbagging.py, fedxgbcyclic.py    # XGBoost
│   ├── qfedavg.py                           # 公平性
│   ├── factory.py                           # 策略工廠
│   └── README.md                # ← 策略詳細說明
│
├── record/                      # 結果記錄 & 可視化
│   ├── recorder.py              # 主控制器
│   ├── data_recorder.py         # 指標收集
│   ├── data_exporter.py         # JSON/CSV 導出
│   ├── draw_acc_chart.py        # 精準度圖表
│   └── README.md                # ← 結果格式說明
│
├── data/                        # 數據集存放
│   └── MNIST/
│       └── raw/                 # MNIST 原始資料
│
├── res/                         # 實驗結果歷史
│   ├── history/                 # 時間戳執行紀錄 (20260322-xxxx)
│   └── IID-Round-200-Epoch-3/   # 歷史對比集
│
├── [三站點資料流程]
│   ├── res-drag-set-to-there----first-station/
│   │   ├── move_outer.py
│   │   ├── generator.py
│   │   ├── check_strategies.py
│   │   └── README.md            # ← First Station 說明
│   │
│   ├── res-happy-ending--------second-station/
│   │   ├── move_outer.py        # 🆕 改為動態命名 + robocopy
│   │   ├── generator.py
│   │   └── README.md            # ← Second Station 說明
│   │
│   └── RESULT_Finally_fan_tasty/
│       └── [最終結果歸檔]
│
├── requirements.txt             # 依賴清單
├── run.py                       # ⭐ 單次執行入口
├── run_selected_strategies.py   # ⭐ 批次執行 18 策略
└── draw_distribution.py         # 數據分布可視化
```

---

## 18️⃣ 聯邦學習策略列表

| 類別 | 策略 | 特性 | 適用場景 |
|------|------|------|--------|
| **基礎** | FedAvg | 標準平均 | 基準對照 |
| | FedAvgM | 動量加速 | 收斂加速 |
| | FedProx | 近端項正則化 | 非 IID 數據 |
| **自適應** | FedAdaGrad | AdaGrad 自適應 | 梯度異質性強 |
| | FedAdam | Adam 自適應 | 梯度異質性強（推薦） |
| | FedYogi | Yogi 自適應 | 梯度異質性強 |
| **Byzantine** | Bulyan | Bulyan 算法 | 抵禦 20% 惡意客戶 |
| | Krum | Krum 聚合 | 抵禦 20% 惡意客戶 |
| | MultiKrum | 多軮 Krum | 抵禦 20% 惡意客戶 |
| **魯棒聚合** | FedMedian | 中位數聚合 | 異常值多 |
| | FedTrimmedAvg | 修剪平均 | 異常值多 |
| **差分隱私** | DP_Client_Fixed | 客戶端 DP（固定σ） | 隱私要求高（嚴格） |
| | DP_Client_Adaptive | 客戶端 DP（自適應σ） | 隱私要求高（靈活） |
| | DP_Server_Fixed | 伺服器 DP（固定σ） | 隱私要求高（集中） |
| | DP_Server_Adaptive | 伺服器 DP（自適應σ） | 隱私要求高（集中靈活） |
| **XGBoost** | FedXgbBagging | 樹模型打包 | 結構化資料 |
| | FedXgbCyclic | 樹模型循環訓練 | 結構化資料 |
| **公平性** | QFedAvg | 次線性公平 | 客戶端參與度不均 |

**⤳ 詳細說明和選擇建議見 [strategies/README.md](strategies/README.md)**

---

## ⚙️ 核心配置參數

所有參數在 **`config/__init__.py`** 中集中管理：

### 策略選擇
```python
STRATEGY_NAME = StrategyName.FEDAVG  # 改為其他策略
```

### 訓練超參
```python
NUM_ROUNDS = 200          # 聯邦訓練輪數
LOCAL_EPOCHS = 3          # 本地訓練輪數
NUM_PARTITIONS = 10       # 客戶端數量
BATCH_SIZE = 32           # 本地批次大小
LEARNING_RATE = 0.01      # 優化器學習率
```

### 數據分布
```python
DATA_DISTRIBUTION = DataDistribution.IID  # IID / LABEL_SKEW / DIRICHLET
```

### 策略特定超參
見 **`config/strategy.py`** 和 **[config/README.md](config/README.md)**

---

## 🔄 三站點資料流程（核心創新）

本項目使用**三站點管道**來組織實驗和對比結果：

```
┌────────────────────┐
│  Raw Results       │
│  (res/history/)    │
└─────────┬──────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ 🚩 FIRST STATION (Drag-Set-To-There)  │
│ ════════════════════════════════════   │
│ 職責：收集、清理、驗證 18 策略的結果 │
│ 輸出：PUT-DATA-THERE/                 │
│ 狀態：✅ 自動化完成                   │
└─────────┬──────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ 🎉 SECOND STATION (Happy-Ending)       │
│ ════════════════════════════════════   │
│ 職責：生成對比圖表、統計分析           │
│ 輸出：generate_charts/                 │
│       (精準度圖、delta 表格 等)       │
│ 狀態：⏳ 待處理（見 0.NEED-*)         │
└─────────┬──────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│ 🏁 FINAL STATION (Finally-Fan-Tasty)   │
│ ════════════════════════════════════   │
│ 職責：歸檔最終結果供報告使用          │
│ 內容：PUT-DATA-THERE/ + generate_* /   │
└────────────────────────────────────────┘
```

**詳細步驟見：**
- [res-drag-set-to-there----first-station/README.md](res-drag-set-to-there----first-station/README.md)
- [res-happy-ending--------second-station/README.md](res-happy-ending--------second-station/README.md)

---

## 📊 執行流程示例

### 場景 1：執行單個策略 + 觀看結果

```bash
# 1. 編輯配置
vim config/__init__.py
# 設置：STRATEGY_NAME = StrategyName.FEDAVG

# 2. 執行訓練
python run.py

# 3. 結果會自動保存到
# res/history/20260325-XXXX-XX/
# ├── run_metrics.csv         # 精準度、損失、輪次
# ├── metadata.txt            # 超參、名稱、耗時
# ├── accuracy_chart.png      # 收斂曲線
# └── distribution_chart.png  # 數據分布
```

### 場景 2：批次執行全部 18 策略 + 對比

```bash
# 1. 確保 config/__init__.py 已配置正確（NUM_ROUNDS, LOCAL_EPOCHS 等）
# 2. 執行批次腳本
python run_selected_strategies.py

# 3. 依序執行所有 18 策略，輸出到 res/history/

# 4. 移動到 First Station 進行清理
cd res-drag-set-to-there----first-station
python move_outer.py  # 自動搬運到 PUT-DATA-THERE/

# 5. 生成對比圖表
cd ../res-happy-ending--------second-station
python generator.py   # 生成圖表到 generate_charts/
python move_outer.py  # 最終搬運到 RESULT_Finally_fan_tasty/

# 6. 查看最終結果
ls ../RESULT_Finally_fan_tasty/
```

---

## 📋 檔案用途快速索引

| 檔案 | 說明 |
|------|------|
| **run.py** | 主入口 - 執行單次聯邦學習模擬 |
| **run_selected_strategies.py** | 批次執行全部 18 策略 |
| **draw_distribution.py** | 繪製客戶端標籤分布直方圖 |
| **config/__init__.py** | ⭐ **全局配置** |
| **app/server.py** | ServerApp：初始化模型、聚合、評估 |
| **app/client.py** | ClientApp：本地訓練、梯度上傳 |
| **app/task.py** | PyTorch CNN 模型 + MNIST 數據加載 |
| **strategies/factory.py** | 策略工廠（註冊 18 種演算法） |

---

## 🐍 Python 版本 & 依賴

```
Python >= 3.9
Requirements:
  - flwr[simulation]==1.25.0     # Federated Learning Framework
  - torch>=2.0                    # PyTorch
  - xgboost>=2.0                  # XGBoost (for FedXgb*)
  - numpy, matplotlib, pandas     # 數據處理 & 可視化
```

---

## 📁 數據分布方式說明

### IID (Independent & Identically Distributed)
- 各客戶端數據完全隨機分割
- **特點**：最理想的聯邦學習場景
- **適合測試**：基礎算法、理論下界

### 標籤傾斜 (Label Skew)
- each client 只有部分類別的數據
- **特點**：高度非均勻分布
- **適合測試**：非 IID 抵禦能力（FedProx, FedAvgM 等）

### Dirichlet (Dir(α))
- 基於 Dirichlet 分布的概率採樣
- **特點**：自然的類別不均勻（α < 1 時）
- **適合測試**：現實場景模擬

---

## 🔍 常見問題

### Q1: 如何執行單一策略？
A: 編輯 `config/__init__.py`，設置 `STRATEGY_NAME`，執行 `python run.py`

### Q2: 如何改變訓練輪數/本地輪數？
A: 編輯 `config/__init__.py` 中的 `NUM_ROUNDS` / `LOCAL_EPOCHS`

### Q3: 如何選擇數據分布？
A: 編輯 `config/__init__.py` 中的 `DATA_DISTRIBUTION`（IID / LABEL_SKEW / DIRICHLET）

### Q4: 結果放在哪裡？
A: `res/history/{timestamp}/` 下，包含 CSV、PNG、metadata.txt

### Q5: 如何對比不同策略？
A: 執行 `run_selected_strategies.py`，再用 First Station & Second Station 管道進行整合對比

### Q6: 為什麼執行會卡在某個策略？
A: 檢查 `config/validation.py` 中的配置驗證；檢查 GPU 記憶體（使用 Ray simulation）

---

## 📚 詳細文檔

- **[strategies/README.md](strategies/README.md)** — 18 個策略詳細說明、選擇建議
- **[config/README.md](config/README.md)** — 配置參數對照表、推薦值
- **[app/README.md](app/README.md)** — ServerApp/ClientApp 執行邏輯
- **[record/README.md](record/README.md)** — 結果輸出格式、CSV 欄位說明
- **[res-drag-set-to-there----first-station/README.md](res-drag-set-to-there----first-station/README.md)** — First Station 使用說明
- **[res-happy-ending--------second-station/README.md](res-happy-ending--------second-station/README.md)** — Second Station 使用說明
- **[data/README.md](data/README.md)** — 數據集、分割策略說明

---

## 📌 最後更新

- **日期**：2026-03-25
- **最近改動**：second-station move_outer.py 改用 robocopy 處理長路徑；資料夾命名改為動態生成（首次改寫）
- **測試環境**：Windows 11 + Python 3.12.4 + Flower 1.25.0

---

## 📞 快速支援

遇到問題？按優先級檢查：
1. 檢查 `config/__init__.py` 配置是否正확
2. 閱讀對應模組的 README.md（見上面詳細文檔）
3. 查看 `res/history/` 下最新執行的 metadata.txt 和日誌
4. 確認 `requirements.txt` 依賴已正確安裝

---

**🎓 基於 Flower 1.25.0 | 聯邦學習研究平台 | 祝好用！**
