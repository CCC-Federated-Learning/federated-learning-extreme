# 🖥️ 應用層 (app/) - Federated Learning 執行引擎

本目錄包含聯邦學習的核心應用邏輯：ServerApp（伺服器）、ClientApp（客戶端）、以及任務定義。

---

## 📁 檔案結構

| 檔案 | 職責 | 關鍵函數 |
|------|------|--------|
| **server.py** | 伺服器應用程式 | `get_server_app()`, `evaluate()` |
| **client.py** | 客戶端應用程式 | `Client` 類, `train()`, `evaluate()` |
| **task.py** | 模型 + 數據加載 | `get_model()`, `load_data()`, `train_one_epoch()` |
| **task_xgb.py** | XGBoost 任務（可選） | `train_xgb()`, `load_xgb_data()` |
| **client_xgb.py** | XGBoost 客戶端（可選） | `XGBClient` 類 |
| **data_cache.py** | 數據集快取 | `cache_dataset()`, `load_cached()` |

---

## 🔄 執行流程（一個聯邦訓練輪）

```
┌─ server.py (ServerApp) ────────────────────────────────────┐
│                                                              │
│  1. 初始化全局模型 w₀                                       │
│     └─ task.get_model()                                     │
│                                                              │
│  2. 發送 w_t 給選中的客戶端                                 │
│                                                              │
│  ├─────────────────┬─────────────────┬─────────────────┐   │
│  │                 │                 │                 │   │
│  ▼                 ▼                 ▼                 ▼   │
│ Client 1       Client 2           Client 3    ...   │
│ (client.py)    (client.py)        (client.py)       │
│                                                       │
│  • 加載本地數據 ──────────────────────────────────┐  │
│    └─ task.load_data() / data_cache.load_cached() │  │
│                                                   │  │
│  • 訓練 LOCAL_EPOCHS 輪 ───────────────────────┐  │  │
│    ├─ task.train_one_epoch()                  │  │  │
│    ├─ (如需) DP 梯度裁剪                      │  │  │
│    └─ 回傳 Δw_t (梯度) 或 w'_t (權重)        │  │  │
│                                               │  │  │
│  (每個客戶端獨立執行，並行運行)              └──┘  │
│                                               ↓     │
│  3. 伺服器聚合客戶端更新 ───────────────────────┐  │
│     ├─ (基礎) 加權平均： w_{t+1} = Σ w'_i      │  │
│     ├─ (Krum) 選距離最近的 k 個                │  │
│     ├─ (DP) 加入高斯噪聲                       │  │
│     └─ (XGBoost) 樹模型聚合                    │  │
│                                               ↓  │
│  4. 評估全局模型 ─────────────────────────────┐  │
│     └─ task.evaluate() (使用評估集)           │  │
│                                               ↓  │
│  5. 輸出本輪指標 ─────────────────────────────┐  │
│     └─ {"loss": ..., "accuracy": ...}         │  │
│                                               ↓  │
│  [重複 NUM_ROUNDS 次]                         │  │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 🔍 詳細模組說明

### **server.py** - ServerApp 伺服器應用程式

#### 職責
- 初始化全局模型
- 在每一輪選擇客戶端
- 聚合客戶端更新
- 評估全局模型
- 記錄指標

#### 關鍵函數

##### `get_server_app() -> ServerApp`
```python
def get_server_app(config: dict) -> ServerApp:
    """建立伺服器應用程式"""
    # 1. 初始化模型
    model = get_model()  # from task.py
    
    # 2. 選擇策略（FedAvg, FedAdam, Krum, 等）
    strategy = factory.get_strategy(STRATEGY_NAME, config)
    
    # 3. 建立 ServerApp
    return ServerApp(
        config=ServerConfig(...),
        strategy=strategy,
        # ... 其他參數
    )
```

##### `evaluate(parameters, config) -> tuple[float, dict]`
```python
def evaluate(parameters, config):
    """評估函數：計算全局模型精準度"""
    # 1. 載入模型 + 參數
    model = get_model()
    model.set_parameters(parameters)
    
    # 2. 在評估集上測試
    test_loader = load_data("test")  # 10% 的數據
    accuracy, loss = test(model, test_loader)
    
    # 3. 回傳結果
    return loss, {"accuracy": accuracy}
```

#### 配置相關程式碼位置
```python
# app/server.py 中
NUM_ROUNDS = config.NUM_ROUNDS              # 聯邦輪數
FRACTION_FIT = 1.0                          # 參與比例（預設全部）
FRACTION_EVALUATE = 1.0                     # 評估客戶端比例
MIN_AVAILABLE_CLIENTS = NUM_PARTITIONS      # 最小可用客戶端
```

---

### **client.py** - ClientApp 客戶端應用程式

#### 職責
- 加載本地數據分片
- 執行本地訓練
- 計算梯度更新（或直接回傳權重）
- 回傳給伺服器

#### 關鍵類

##### `Client` 類
```python
class Client(NumPyClient):
    def __init__(self, cid: str, partition):
        """初始化客戶端
        
        Args:
            cid: 客戶端 ID (0~NUM_PARTITIONS-1)
            partition: 本地數據（由 task.load_data 分割）
        """
        self.cid = cid
        self.partition = partition  # (X_train, y_train, X_val, y_val)
```

##### `fit(parameters, config) -> tuple[parameters, int, dict]`
```python
def fit(self, parameters, config):
    """本地訓練函數（每輪被伺服器調用）
    
    Returns:
        parameters: 訓練後的權重
        num_examples: 使用的樣本數
        metrics: 本地訓練指標 {"loss": ..., "accuracy": ...}
    """
    # 1. 加載本地模型
    model = get_model()
    model.set_parameters(parameters)  # 接收伺服器的權重
    
    # 2. 訓練 LOCAL_EPOCHS 輪
    for epoch in range(config.local_epochs):
        task.train_one_epoch(model, self.partition["train"], config)
    
    # 3. (可選) DP 梯度裁剪
    if config.dp_enabled:
        # 裁剪梯度到 clipping_norm
        # 加入高斯噪聲
        pass
    
    # 4. 回傳更新後的權重 + 樣本數
    return model.get_parameters(), len(self.partition["train"]), {}
```

##### `evaluate(parameters, config) -> tuple[float, int, dict]`
```python
def evaluate(self, parameters, config):
    """評估客戶端本地模型（可選，通常用於診斷）"""
    model = get_model()
    model.set_parameters(parameters)
    
    # 在本地驗證集上測試
    val_loss, val_acc = task.evaluate(model, self.partition["val"])
    
    return val_loss, len(self.partition["val"]), {"accuracy": val_acc}
```

#### 客戶端數據分割
```python
# 在 task.py 中
X_train, y_train, X_test, y_test = load_data()
partitions = create_partitions(X_train, y_train, NUM_PARTITIONS, DATA_DISTRIBUTION)

# partitions[i] = {
#     "train": (X_i, y_i),
#     "val": (X_val_i, y_val_i)
# }

# 詳見下面 task.py 說明
```

---

### **task.py** - 模型 + 數據加載

#### 職責
- 定義 PyTorch 模型
- 加載和分割 MNIST 數據
- 提供訓練和評估函數

#### 關鍵函數

##### `get_model() -> torch.nn.Module`
```python
def get_model():
    """返回 PyTorch 模型（CNN for MNIST）"""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=5),      # 1×28×28 → 16×24×24
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),                    # → 16×12×12
        
        torch.nn.Conv2d(16, 32, kernel_size=5),     # → 32×8×8
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),                    # → 32×4×4
        
        torch.nn.Flatten(),                          # → 512
        torch.nn.Linear(512, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)                      # → 10 (類別數)
    )

# 模型參數數量：~35K（輕量級）
```

##### `load_data() -> tuple[X, y, X_test, y_test]`
```python
def load_data():
    """加載 MNIST 數據集"""
    # 1. 下載 MNIST（如果需要）
    trainset = torch.utils.data.MNIST("data/MNIST/raw", download=True)
    testset = torch.utils.data.MNIST("data/MNIST/raw", train=False)
    
    # 2. 轉換為 NumPy
    X_train = trainset.data.numpy() / 255.0  # 正規化到 [0,1]
    y_train = trainset.targets.numpy()
    
    X_test = testset.data.numpy() / 255.0
    y_test = testset.targets.numpy()
    
    return X_train, y_train, X_test, y_test

# 返回值形狀：
#   X_train: (60000, 28, 28)
#   y_train: (60000,)
#   X_test: (10000, 28, 28)
#   y_test: (10000,)
```

##### `create_partitions(X, y, num_partitions, distribution) -> list[dict]`
```python
def create_partitions(X, y, num_partitions, distribution):
    """將訓練集分割給 NUM_PARTITIONS 個客戶端"""
    
    if distribution == DataDistribution.IID:
        # 完全隨機分割
        indices = np.random.permutation(len(X))
        
    elif distribution == DataDistribution.LABEL_SKEW:
        # 每個客戶端只有部分類別
        # 客戶端 0: 類別 0-4
        # 客戶端 1: 類別 5-9
        indices = partition_by_labels(y, num_partitions)
        
    elif distribution == DataDistribution.DIRICHLET:
        # Dirichlet(α) 採樣
        # α < 1: 高度非均勻
        # α = 1: 中等非均勻
        # α >> 1: 接近 IID
        indices = partition_by_dirichlet(y, num_partitions, alpha=0.5)
    
    # 分割成客戶端數據
    partitions = []
    for i in range(num_partitions):
        idx = indices[client_idx]
        X_i, y_i = X[idx], y[idx]
        
        # 進一步分為訓練和驗證
        X_train_i, X_val_i = train_test_split(X_i, test_size=0.2)
        y_train_i, y_val_i = train_test_split(y_i, test_size=0.2)
        
        partitions.append({
            "train": (X_train_i, y_train_i),
            "val": (X_val_i, y_val_i)
        })
    
    return partitions
```

##### `train_one_epoch(model, train_data, config) -> dict`
```python
def train_one_epoch(model, train_data, config):
    """訓練一個 epoch"""
    X, y = train_data
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # 建立 DataLoader
    dataset = MNIST_Dataset(X, y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )
    
    # 訓練迴圈
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return {"train_loss": total_loss / len(loader)}
```

##### `evaluate(model, test_data) -> tuple[float, float]`
```python
def evaluate(model, test_data):
    """評估模型精準度和損失"""
    X, y = test_data
    dataset = MNIST_Dataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256)
    
    model.eval()
    total_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
    
    accuracy = correct / len(y)
    avg_loss = total_loss / len(loader)
    
    return avg_loss, accuracy
```

---

### **task_xgb.py** & **client_xgb.py** - XGBoost 支持

用於結構化數據（表格）的聯邦 XGBoost 訓練。詳見 [strategies/README.md](../strategies/README.md) 中 FedXgbBagging/FedXgbCyclic 部分。

---

### **data_cache.py** - 數據快取

優化多輪訓練的效率，避免重複加載數據：

```python
def cache_dataset():
    """首次執行時快取數據集"""
    X_train, y_train, X_test, y_test = load_data()
    partitions = create_partitions(X_train, y_train, NUM_PARTITIONS, DATA_DISTRIBUTION)
    
    # 保存到快取檔案（pickle）
    pickle.dump(partitions, open(".cached_partitions.pkl", "wb"))

def load_cached(num_partitions):
    """加載快取數據（加快後續執行）"""
    if os.path.exists(".cached_partitions.pkl"):
        return pickle.load(open(".cached_partitions.pkl", "rb"))
    else:
        return cache_dataset()
```

---

## 🔐 差分隱私 (DP) 流程

若選擇 DP 策略（DP_Client_* 或 DP_Server_*），客戶端會執行以下額外步驟：

```python
# 在 client.py 的 fit() 中

# 1. 訓練後取得梯度（或權重差）
gradients = compute_gradients(model, before_params)

# 2. 梯度裁剪（L2 norm）
clipped_gradients = clip_l2_norm(gradients, config.clipping_norm)

# 3. 加入高斯噪聲
noise = np.random.normal(0, config.noise_multiplier * config.clipping_norm, 
                          size=gradients.shape)
noisy_gradients = clipped_gradients + noise

# 4. 回傳雜訊梯度
return noisy_gradients
```

---

## 🚀 執行範例

### 單次執行（一個聯邦輪）

```python
# app/server.py 中調用

from app.server import get_server_app
from app.client import client_fn

# 1. 創建 ServerApp
server_app = get_server_app(config)

# 2. 創建客戶端工廠函數
def client_fn(cid: str) -> Client:
    partition = partitions[int(cid)]
    return Client(cid, partition)

# 3. 啟動模擬（由 Flower 框架自動執行）
# Flower 會調用：
#   - client_fn("0"), client_fn("1"), ... 來創建客戶端
#   - client.fit() 進行本地訓練
#   - server.aggregate() 進行聚合
#   - server.evaluate() 進行評估
#   - 重複 NUM_ROUNDS 次
```

---

## 📊 輸出指標

每一輪聯邦訓練結束後，`server.py` 會輸出：

```
[ROUND 1/200]
├─ Global Model Loss: 2.301
├─ Global Model Accuracy: 0.098
├─ Number of Clients Sampled: 10
└─ Time: 12.5s

[ROUND 2/200]
├─ Global Model Loss: 2.251
├─ Global Model Accuracy: 0.156
├─ Number of Clients Sampled: 10
└─ Time: 11.8s

... (重複 200 次)

[FINAL RESULTS]
├─ Final Accuracy: 0.956
├─ Final Loss: 0.142
└─ Total Training Time: 2345.6s
```

---

## 🔧 常見問題

### Q1: 如何修改模型架構？
A: 編輯 `task.py` 中的 `get_model()` 函數，或自定義一個新模型類。

### Q2: 如何使用自己的數據集？
A: 編輯 `task.py` 中的 `load_data()` 函數，加載你的數據（確保回傳格式相同）。

### Q3: 如何加速訓練？
A: 
- 提高 `BATCH_SIZE`（有 GPU 時）
- 降低 `NUM_ROUNDS` 或 `LOCAL_EPOCHS`
- 使用 `data_cache.py` 快取數據

### Q4: 為什麼客戶端訓練這麼慢？
A: 檢查：
- Local epochs 是否過多
- Batch size 是否過小
- 模型是否太大

---

**了解更多**：見 [app/ 原始碼](.) 或根目錄 [README.md](../README.md)
