# 📊 18 個聯邦學習策略詳細說明

本項目實現了 18 種聯邦學習策略，涵蓋基礎平均、自適應優化、拜占庭防護、差分隱私、XGBoost 訓練、公平性優化等維度。

---

## 🎯 策略分類速查表

| 類別 | 策略名 | Python 類別 | 檔案 | 難度 | 推薦指數 |
|------|--------|-----------|------|------|--------|
| **基礎層** | FedAvg | `FedAvg` | `fedavg.py` | ⭐ | ⭐⭐⭐⭐⭐ |
| | FedAvgM | `FedAvgM` | `fedavgm.py` | ⭐⭐ | ⭐⭐⭐⭐ |
| | FedProx | `FedProx` | `fedprox.py` | ⭐⭐ | ⭐⭐⭐⭐ |
| **自適應層** | FedAdaGrad | `FedAdaGrad` | `fedadagrad.py` | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| | FedAdam | `FedAdam` | `fedadam.py` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| | FedYogi | `FedYogi` | `fedyogi.py` | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Byzantine** | Bulyan | `Bulyan` | `bulyan.py` | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| | Krum | `Krum` | `krum.py` | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| | MultiKrum | `MultiKrum` | `multikrum.py` | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **魯棒聚合** | FedMedian | `FedMedian` | `fedmedian.py` | ⭐⭐ | ⭐⭐⭐ |
| | FedTrimmedAvg | `FedTrimmedAvg` | `fedtrimmedavg.py` | ⭐⭐ | ⭐⭐⭐⭐ |
| **差分隱私** | DP_Client_Fixed | `DPClientFixed` | `dp_client_fixed.py` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| | DP_Client_Adaptive | `DPClientAdaptive` | `dp_client_adaptive.py` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| | DP_Server_Fixed | `DPServerFixed` | `dp_server_fixed.py` | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| | DP_Server_Adaptive | `DPServerAdaptive` | `dp_server_adaptive.py` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **XGBoost** | FedXgbBagging | `FedXgbBagging` | `fedxgbbagging.py` | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| | FedXgbCyclic | `FedXgbCyclic` | `fedxgbcyclic.py` | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **公平性** | QFedAvg | `QFedAvg` | `qfedavg.py` | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 📖 策略詳細說明

### 🔷 基礎層（Base Layer）

#### 1. **FedAvg** - Federated Averaging
- **原論文**：McMahan et al., 2017
- **核心思想**：簡單加權平均客戶端更新
- **公式**：$w_{t+1} = \sum_i \frac{n_i}{n} w_i^t$
- **優點**：
  - 簡單、穩定、通用
  - 理論下界（optimal convergence）
  - 零隱私成本
- **缺點**：
  - 非 IID 數據下收斂慢
  - 無自適應學習率
- **适用場景**：
  - ✅ IID 數據分佈
  - ✅ 基準對照（baseline）
  - ✅ 教學演示
- **推薦超參**：
  ```python
  # 默認即可，無額外超參
  NUM_ROUNDS = 200
  LOCAL_EPOCHS = 3
  LEARNING_RATE = 0.01
  ```

---

#### 2. **FedAvgM** - Federated Averaging with Momentum
- **原論文**：Wang et al., 2021
- **核心思想**：在伺服器端應用動量加速
- **公式**：$v_{t+1} = \beta v_t + (1-\beta) \Delta w_t$，$w_{t+1} = w_t - \eta v_{t+1}$
- **優點**：
  - 加速收斂（尤其非 IID 數據）
  - 簡單易實現
  - 無隱私損失
- **缺點**：
  - 引入額外超參 β（動量係數）
  - 不如自適應方法穩定
- **适用場景**：
  - ✅ IID 和輕度非 IID
  - ✅ 需要快速收斂的場景
  - ✅ 已知較好的學習率
- **推薦超參**：
  ```python
  SERVER_MOMENTUM = 0.9  # 動量係數（config/strategy.py）
  NUM_ROUNDS = 150-200
  LOCAL_EPOCHS = 3
  ```

---

#### 3. **FedProx** - Federated Proximal
- **原論文**：Li et al., 2020
- **核心思想**：伺服器端加入近端正則化項，抵抗本地 drift
- **公式**：$w_{t+1} = \arg\min_w \frac{1}{n}\sum_i ||w - w_i^t||^2 + \mu L(w)$
- **優點**：
  - 對非 IID 數據友好
  - 收斂有保證（即使本地步數不同）
- **缺點**：
  - 引入超參 μ（近端係數）
  - 計算量略增
- **适用場景**：
  - ✅ **高度非 IID 數據（推薦）**
  - ✅ 客戶端計算能力異質（本地 epochs 不同）
  - ✅ 標籤傾斜分布
- **推薦超參**：
  ```python
  PROXIMAL_MU = 0.01 ~ 0.1  # 根據非 IID 程度調整（config/strategy.py）
  NUM_ROUNDS = 200
  LOCAL_EPOCHS = 3-5
  ```

---

### 🟢 自適應層（Adaptive Layer）

#### 4. **FedAdaGrad** - Federated AdaGrad
- **核心思想**：伺服器端應用 AdaGrad，維護各參數的累積梯度平方和
- **公式**：$g_{t+1} = g_t + (\Delta w_t)^2$，$w_{t+1} = w_t - \eta / \sqrt{g_{t+1}} \Delta w_t$
- **優點**：
  - 自動調整學習率
  - 對梯度異質性敏感度强
- **缺點**：
  - 學習率單調遞減（可能過早停滯）
  - 需要較大初始學習率
- **适用場景**：
  - ✅ 梯度異質性強
  - ✅ 參數更新幅度差異大
- **推薦超參**：
  ```python
  NUM_ROUNDS = 200
  LOCAL_EPOCHS = 3
  LEARNING_RATE = 0.05 ~ 0.1  # 較大初值
  ```

---

#### 5. **FedAdam** - Federated Adam（⭐ 最推薦）
- **核心思想**：伺服器端應用 Adam 優化器，同時維護一階矩和二階矩
- **公式**：
  - $m_t = \beta_1 m_{t-1} + (1-\beta_1) \Delta w_t$
  - $v_t = \beta_2 v_{t-1} + (1-\beta_2) (\Delta w_t)^2$
  - $w_{t+1} = w_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$
- **優點**：
  - 🌟 **最穩定、最推薦**
  - 學習率自動調整
  - 動量 + 自適應雙重加速
  - 對各種數據分布都友好
- **缺點**：
  - 超參較多（β₁, β₂, ε）
  - 計算稍複雜
- **适用場景**：
  - ✅ **通用（優先選擇）**
  - ✅ 不確定最佳超參時
  - ✅ 任何數據分布
- **推薦超參**：
  ```python
  SERVER_MOMENTUM = 0.9         # β₁
  SERVER_MOMENTUM2 = 0.99       # β₂
  NUM_ROUNDS = 150-200
  LOCAL_EPOCHS = 3
  LEARNING_RATE = 0.01 ~ 0.05
  ```

---

#### 6. **FedYogi** - Federated Yogi
- **核心思想**：Yogi 優化器的聯邦版本，二階矩更新方式不同
- **公式**：$v_t = v_{t-1} - (1-\beta_2) \text{sign}(v_{t-1} - (\Delta w_t)^2) \odot (\Delta w_t)^2$
- **優點**：
  - 比 Adam 更激進的學習率調整
  - 快速適應梯度變化
- **缺點**：
  - 可能震蕩（高方差數據）
  - 超參更難調
- **适用場景**：
  - ✅ 梯度方差低
  - ✅ 需要快速適應
- **推薦超參**：
  ```python
  SERVER_MOMENTUM = 0.9
  NUM_ROUNDS = 150-200
  LOCAL_EPOCHS = 3
  ```

---

### 🔴 Byzantine 防護層（Byzantine Robust）

#### 7. **Bulyan**
- **核心思想**：多層過濾，先用 Krum 聚合，再用中位數+修剪
- **防護能力**：⭐⭐⭐⭐⭐ 最強
- **耐受惡意比例**：~25% 惡意客戶端
- **優點**：
  - 最高的拜占庭防護
  - 對各種攻擊都有防禦
- **缺點**：
  - 計算複雜度最高
  - 會削弱正常表現（overly conservative）
- **适用場景**：
  - ✅ 不信任客戶端
  - ✅ 高安全要求
  - ✅ 已知可能有攻擊
- **推薦超參**：
  ```python
  NUM_ROUNDS = 300-400  # 需要更多輪以補償保守性
  LOCAL_EPOCHS = 3
  ```

---

#### 8. **Krum**
- **核心思想**：選擇距離鄰近的 k 個客戶端梯度进行平均
- **耐受惡意比例**：~20% 惡意客戶端
- **優點**：
  - 計算量相對小
  - 性能下降不太多
  - 易於理解
- **缺點**：
  - 不如 Bulyan 全面
- **适用場景**：
  - ✅ 適度安全需求
  - ✅ 計算資源有限
- **推薦超參**：
  ```python
  NUM_ROUNDS = 200-250
  LOCAL_EPOCHS = 3
  ```

---

#### 9. **MultiKrum**
- **核心思想**：執行多輪 Krum，取 k 個最好的聚合結果
- **耐受惡意比例**：~20% 惡意客戶端
- **優點**：
  - 比單輪 Krum 更穩定
  - 保留更多資訊
- **缺點**：
  - 計算量增加
- **适用場景**：
  - ✅ 中等安全需求
  - ✅ 數據足夠大

---

### 💛 魯棒聚合層（Robust Aggregation）

#### 10. **FedMedian**
- **核心思想**：使用中位數代替平均值
- **優點**：
  - 對異常梯度鈍感
  - 簡單易實現
- **缺點**：
  - 信息損失（只取中位，拋棄其他）
  - 不如專門的拜占庭算法
- **适用場景**：
  - ✅ 少量異常梯度
  - ✅ 簡單場景

---

#### 11. **FedTrimmedAvg**
- **核心思想**：去掉最大和最小的 β% 梯度，對剩餘的求平均
- **優點**：
  - 平衡異常處理和信息保留
  - 靈活性好（可調 β）
- **缺點**：
  - 需要選擇合適的 β
- **适用場景**：
  - ✅ 中等異常值
  - ✅ 數據分布有偏離
- **推薦超參**：
  ```python
  TRIM_RATIO = 0.1 ~ 0.2  # 去掉 10-20%
  ```

---

### 🔐 差分隱私層（Differential Privacy）

本項目提供 **4 種 DP 變體**，涵蓋客戶端和伺服器端隱私保護：

#### 12 & 13. **DP_Client_Fixed & DP_Client_Adaptive**
- **應用位置**：梯度上傳前，客戶端加入 Gaussian 噪聲
- **流程**：
  1. 客戶端計算梯度
  2. 單個梯度裁剪（L2 norm ≤ clipping threshold）
  3. 加入高斯噪聲 $\mathcal{N}(0, \sigma^2)$
  4. 上傳至伺服器聚合

#### Fixed vs Adaptive
| 特性 | Fixed | Adaptive |
|------|-------|----------|
| σ（噪聲強度） | 固定 | 動態調整 |
| 隱私成本 | 均勻 | 非均勻（靈活） |
| 推薦場景 | 簡單場景 | **複雜場景（推薦）** |
| 超參調整 | 容易 | 較難 |

- **優點**：
  - 個別客戶端隱私最強
  - 保護最敏感的信息
- **缺點**：
  - 噪聲大，模型性能下降
  - 需調整噪聲倍數平衡隱私-效用
- **推薦超參**：
  ```python
  DP_NOISE_MULTIPLIER = 0.1 ~ 1.0  # 噪聲強度（config/strategy.py）
                                     # 小 = 性能好但隱私弱
                                     # 大 = 隱私強但性能下降
  DP_CLIPPING_NORM = 1.0            # 梯度裁剪閾值
  NUM_ROUNDS = 300-500              # 需要更多輪補償噪聲
  ```

---

#### 14 & 15. **DP_Server_Fixed & DP_Server_Adaptive**
- **應用位置**：伺服器端聚合後加入噪聲
- **流程**：
  1. 伺服器接收客戶端梯度（無雜訊）
  2. 聚合為全局梯度
  3. 加入高斯噪聲
  4. 更新全局模型

#### Fixed vs Adaptive
同上表

- **優點**：
  - 性能相對保留較好（客戶端無噪聲）
  - 實現較簡單
- **缺點**：
  - 隱私保護不如客戶端端到端
  - 需信任伺服器
- **推薦超參**：
  ```python
  DP_NOISE_MULTIPLIER = 0.01 ~ 0.5  # 伺服器端噪聲可以更小
  NUM_ROUNDS = 200-300
  ```

---

#### ✅ 何時使用 DP？

| 場景 | 推薦 |
|------|------|
| 金融機構（高隱私要求） | ✅ DP_Client_Adaptive |
| 醫療數據（HIPAA 合規） | ✅ DP_Client_Adaptive |
| 簡單演示（教學） | ❌ 不用 DP |
| 已有加密通道（信任伺服器） | ✅ DP_Server_Adaptive |

---

### 🌳 XGBoost 聯邦訓練層

#### 16 & 17. **FedXgbBagging & FedXgbCyclic**

**何時使用 XGBoost**：
- 數據**結構化**（表格）而非高維圖片
- 需要**可解釋性**
- 數據量**中等**（不適合海量）

#### Bagging vs Cyclic
| 特性 | Bagging | Cyclic |
|------|---------|--------|
| 構建方式 | 隨機採樣子集構建樹 | 循環遍歷所有特徵 |
| 並行度 | 高（各樹獨立） | 低（順序依賴） |
| 性能 | 通常更好 | 可能較差 |
| 推薦度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

- **優點**：
  - 結構化數據上表現優異
  - 可解釋性強
  - 無隱私成本
- **缺點**：
  - 僅支持結構化數據
  - 實現複雜
- **推薦超參**：
  ```python
  XGB_ETA = 0.3             # 學習率（config/strategy.py）
  XGB_MAX_DEPTH = 6         # 樹深度
  NUM_ROUNDS = 50-100       # XGBoost 輪數通常較少
  ```

---

### 🎯 公平性優化層

#### 18. **QFedAvg** - Fairness-Aware Federated Averaging
- **原論文**：Hsu et al., 2019
- **核心思想**：次線性公平函數，確保慢速客戶端也能進步
- **公式**：$\min_w \sum_i q_i L_i(w)$，其中 $q_i \propto (L_i)^{-1/t}$
- **優點**：
  - 確保參與度低的客戶端也能改進
  - 適應異質化硬體
- **缺點**：
  - 計算権重複雜
  - 可能拖累整體性能
- **适用場景**：
  - ✅ 客戶端計算能力差異大
  - ✅ 強調公平性而非效率
- **推薦超參**：
  ```python
  NUM_ROUNDS = 200
  LOCAL_EPOCHS = 3-5
  ```

---

## 🏆 策略選擇建議

### 按場景推薦

| 場景 | 最佳選擇 | 備選 |
|------|---------|------|
| **IID 數據 + 快速演示** | FedAvg | FedAvgM |
| **非 IID 數據 + 生產環境** | FedAdam | FedProx |
| **高隱私要求** | DP_Client_Adaptive | DP_Server_Adaptive |
| **不信任客戶端** | Bulyan | Krum |
| **結構化數據（表格）** | FedXgbBagging | FedXgbCyclic |
| **客戶端異質性強** | QFedAvg | FedAdam |

---

### 按優先級排序

```
第 1 優先級（必須了解）：
  1. FedAvg      - 基準
  2. FedAdam     - 最穩定通用
  3. FedProx     - 非 IID 專家

第 2 優先級（常用）：
  4. FedAvgM     - 快速收斂
  5. Krum        - 拜占庭防護
  6. FedTrimmedAvg - 魯棒聚合

第 3 優先級（特殊用途）：
  7~18. 其他策略  - 按需選擇
```

---

## 📝 如何使用

### 方式 1：在代碼中選擇
```python
# config/__init__.py

from config.types import StrategyName

STRATEGY_NAME = StrategyName.FEDADAM  # 改這行
```

### 方式 2：查看所有可用策略
```python
python -c "from config.types import StrategyName; print([s.name for s in StrategyName])"
```

輸出：
```
['FEDAVG', 'FEDAVGM', 'FEDPROX', 'FEDADAGRAD', 'FEDADAM', 'FEDYOGI', 
 'BULYAN', 'KRUM', 'MULTIKRUM', 'FEDMEDIAN', 'FEDTRIMMEDAVG', 
 'DP_CLIENT_FIXED', 'DP_CLIENT_ADAPTIVE', 'DP_SERVER_FIXED', 'DP_SERVER_ADAPTIVE', 
 'FEDXGBBAGGING', 'FEDXGBCYCLIC', 'QFEDAVG']
```

---

## 🔧 策略工廠（Factory Pattern）

所有策略都通過 **`strategies/factory.py`** 註冊和實例化：

```python
# strategies/factory.py
def get_strategy(strategy_name: StrategyName, config: dict):
    strategies = {
        StrategyName.FEDAVG: FedAvg,
        StrategyName.FEDADAM: FedAdam,
        # ... 等等 18 個
    }
    return strategies[strategy_name](**config)
```

---

## 📚 參考論文

| 策略 | 原論文 | 年份 |
|------|--------|------|
| FedAvg | McMahan et al. | 2017 |
| FedAvgM | Wang et al. | 2021 |
| FedProx | Li et al. | 2020 |
| FedAdam | Reddi et al. | 2021 |
| Bulyan | Yin et al. | 2018 |
| Krum | Blanchard et al. | 2017 |
| QFedAvg | Hsu et al. | 2019 |
| Diff Privacy | Kairouz et al. | 2021 |

---

## 💡 最後提示

1. **優先選 FedAdam**（如果不確定）
2. **對非 IID 數據用 FedProx**
3. **隱私要求用 DP_Client_Adaptive**
4. **需要體驗拜占庭用 Krum**
5. **表格數據用 FedXgbBagging**

祝你實驗順利！ 🚀
