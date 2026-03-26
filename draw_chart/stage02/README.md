# 🎉 Second Station: 可視化對比與整理

**res-happy-ending--------second-station** 是三站點流程中的**第二關口**，負責從 First Station 的清理結果中生成精美的對比圖表和統計分析。

---

## 🎯 職責一覽

| 步驟 | 檔案 | 什麼做 |
|------|------|------|
| 1️⃣ **生成圖表** | `generator.py` | 綜合 18 個策略的結果，生成多層次對比圖 |
| 2️⃣ **整理搬運** | `move_outer.py` | 將結果組織並搬至 Final Station |

---

## 📊 輸入與輸出

### 輸入
```
../res-drag-set-to-there----first-station/PUT-DATA-THERE/
├── IID-Round-20-Epoch-1-20260324-2043-57/
│   ├── run_metrics.csv
│   └── metadata.txt
├── label-Round-20-Epoch-1-20260324-2232-26/
└── ...
```

### 輸出
```
generate_charts/
├── 01_all_18x2_acc_round_iid_vs_label.png           # 18 策略全覆蓋
├── 02_grouped_acc_round_iid_vs_label.png            # 按類別分組
├── 03a_table_final_acc_delta_grouped.png            # 精準度 delta 表（分組）
├── 03b_table_final_acc_delta_sorted.png             # 精準度 delta 表（排序）
├── 03c_table_final_acc_delta_grouped_by_category... # 按類別分組的 delta
├── 04a_bar_chart_sorted_by_delta.png                # 條形圖（按 delta 排序）
├── 04b_bar_chart_sorted_by_iid.png                  # 條形圖（按 IID 精準度）
├── 04c_bar_chart_sorted_by_label.png                # 條形圖（按 Label 精準度）
└── per_strategy_iid_vs_label/                        # 單個策略的對比
    ├── FedAvg_iid_vs_label.png
    ├── FedAdam_iid_vs_label.png
    └── ...（18 個策略各一個）
```

---

## 🔧 使用步驟

### 步驟 1：從 First Station 獲得清理結果

確保 First Station 已完成：
```bash
ls ../res-drag-set-to-there----first-station/PUT-DATA-THERE/
# 應該看到多個 {distribution}-Round-... 資料夾
```

---

### 步驟 2：轉到 Second Station

```bash
cd res-happy-ending--------second-station
```

---

### 步驟 3：生成圖表

```bash
# 方式 1: 只生成圖表（只讀 First Station 的結果）
python generator.py

# 方式 2: 生成圖表 + 搬運到 Final Station
python move_to_stage03.py
```

**預期輸出**：
```
======================================================================
Second-Station Results Organization Script
======================================================================

✓ Found PUT-DATA-THERE directory
✓ Found generate_charts directory

Extracting experiment configuration...
  Data Distribution: IID-label
  Rounds: 20
  Local Epochs: 1
  Run ID: 20260324-2043-57

📁 Creating directory: IID-label-Round-20-Epoch-1-20260324-2043-57
✓ Ensured RESULT_Finally_fan_tasty directory exists
✓ Created target directory: ...

Moving PUT-DATA-THERE...
✓ Copied PUT-DATA-THERE

Moving generate_charts...
✓ Copied generate_charts
✓ Created SUMMARY.txt

======================================================================
✅ SUCCESS: Results organized and moved!
======================================================================
```

---

## 📖 詳細說明

### **generator.py** - 圖表生成器

#### 職責
- 讀取 First Station 的所有結果
- 生成多層次的對比圖表
- 計算策略之間的性能差異

#### 關鍵函數

##### `load_all_results(put_data_dir) -> dict`
```python
def load_all_results(put_data_dir):
    """讀取 First Station 所有結果"""
    results = {}
    
    # 掃描每個 {distribution}-Round-... 資料夾
    for result_dir in sorted(put_data_dir.glob("*")):
        csv_file = result_dir / "run_metrics.csv"
        metadata_file = result_dir / "metadata.txt"
        
        if csv_file.exists() and metadata_file.exists():
            # 讀取數據
            df = pd.read_csv(csv_file)
            metadata = read_metadata(metadata_file)
            
            # 提取關鍵信息
            key = (metadata['data_distribution'], metadata['strategy_name'])
            results[key] = {
                'accuracy': df['accuracy'].values,
                'loss': df['loss'].values,
                'final_accuracy': df['accuracy'].iloc[-1],
                'final_loss': df['loss'].iloc[-1],
                'metadata': metadata
            }
    
    return results
```

##### `plot_all_strategies() -> None`
```python
def plot_all_strategies(results, output_dir):
    """
    生成：01_all_18x2_acc_round_iid_vs_label.png
    展示：18 個策略在 IID 和 Label 兩種分布下的精準度曲線
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 18))  # 18 種顏色
    
    # IID 曲線
    for i, strategy in enumerate(STRATEGY_NAMES):
        if ('IID', strategy) in results:
            acc_curve = results[('IID', strategy)]['accuracy']
            ax1.plot(range(len(acc_curve)), acc_curve, 
                    label=strategy, color=colors[i], linewidth=1.5)
    
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("All 18 Strategies - IID Distribution")
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Label 曲線（重複上述，改為 LABEL_SKEW）
    for i, strategy in enumerate(STRATEGY_NAMES):
        if ('LABEL_SKEW', strategy) in results:
            acc_curve = results[('LABEL_SKEW', strategy)]['accuracy']
            ax2.plot(range(len(acc_curve)), acc_curve, 
                    label=strategy, color=colors[i], linewidth=1.5)
    
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("All 18 Strategies - LABEL_SKEW Distribution")
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "01_all_18x2_acc_round_iid_vs_label.png", dpi=150)
    plt.close()
```

##### `plot_grouped_strategies() -> None`
```python
def plot_grouped_strategies(results, output_dir):
    """
    生成：02_grouped_acc_round_iid_vs_label.png
    展示：按策略類別（Basic/Adaptive/Byzantine/DP/XGBoost/Fair）分組的曲線
    """
    groups = {
        'Basic': ['FedAvg', 'FedAvgM', 'FedProx'],
        'Adaptive': ['FedAdaGrad', 'FedAdam', 'FedYogi'],
        'Byzantine': ['Bulyan', 'Krum', 'MultiKrum'],
        'Robust': ['FedMedian', 'FedTrimmedAvg'],
        'Differential Privacy': ['DP_ClientFixed', 'DP_ClientAdaptive', 'DP_ServerFixed', 'DP_ServerAdaptive'],
        'XGBoost': ['FedXgbBagging', 'FedXgbCyclic'],
        'Fairness': ['QFedAvg']
    }
    
    # 為每個組繪製子圖…
    # ([實現代碼] …)
```

##### `plot_delta_table() -> None`
```python
def plot_delta_table(results, output_dir):
    """
    生成：03a_table_final_acc_delta_grouped.png 等
    展示：策略性能 delta 表（IID vs Label 的精準度差異）
    """
    table_data = []
    
    for strategy in STRATEGY_NAMES:
        iid_acc = results.get(('IID', strategy), {}).get('final_accuracy', np.nan)
        label_acc = results.get(('LABEL_SKEW', strategy), {}).get('final_accuracy', np.nan)
        
        delta = iid_acc - label_acc  # 非 IID 時的性能下降
        
        table_data.append({
            'Strategy': strategy,
            'IID Acc': f"{iid_acc:.4f}",
            'Label Acc': f"{label_acc:.4f}",
            'Delta (↓)': f"{delta:.4f}"
        })
    
    # 排序並繪製表格
    df_table = pd.DataFrame(table_data).sort_values('Delta (↓)', ascending=False)
    
    # 使用 matplotlib 繪製表格為 PNG
    fig, ax = plt.subplots(figsize=(10, 8))
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax.axis('off')
    
    plt.savefig(output_dir / "03a_table_final_acc_delta_grouped.png", dpi=150, bbox_inches='tight')
    plt.close()
```

---

### **move_outer.py** - 搬運整理器（改寫版）

**最新版本**（已支援 robocopy 長路徑）：

```python
def aggregate_results():
    """搬運結果到 Final Station"""
    
    # 1. 檢查 PUT-DATA-THERE 和 generate_charts
    put_data_dir = Path("PUT-DATA-THERE")
    generate_charts_dir = Path("generate_charts")
    
    if not put_data_dir.exists() or not generate_charts_dir.exists():
        print("❌ Error: Missing PUT-DATA-THERE or generate_charts")
        return False
    
    # 2. 從 PUT-DATA-THERE 提取配置
    config = get_experiment_config(put_data_dir)
    
    # 3. 動態生成資料夾名稱
    # 格式：{distribution}-Round-{rounds}-Epoch-{epochs}-{run_id}
    output_dir_name = create_directory_name(config)
    
    # 4. 在 Final Station 建立資料夾
    final_station = Path("../") / "RESULT_Finally_fan_tasty"
    target_dir = final_station / output_dir_name
    
    if target_dir.exists():
        print(f"⚠️  Directory already exists: {target_dir}")
        print("Removing existing directory...")
        shutil.rmtree(target_dir)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. 複製 PUT-DATA-THERE（使用 robocopy 處理長路徑）
    print("\nMoving PUT-DATA-THERE...")
    result = subprocess.run(
        ['robocopy', str(put_data_dir), str(target_dir / 'PUT-DATA-THERE'), '/E', '/MT:4'],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode >= 8:
        print(f"❌ Error copying PUT-DATA-THERE")
        return False
    print("✓ Copied PUT-DATA-THERE")
    
    # 6. 複製 generate_charts（使用 robocopy）
    print("\nMoving generate_charts...")
    result = subprocess.run(
        ['robocopy', str(generate_charts_dir), str(target_dir / 'generate_charts'), '/E', '/MT:4'],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode >= 8:
        print(f"❌ Error copying generate_charts")
        return False
    print("✓ Copied generate_charts")
    
    # 7. 生成 SUMMARY.txt
    summary_file = target_dir / 'SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SECOND-STATION RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        # ... 寫入摘要信息
    
    print(f"\n✅ SUCCESS: Results organized!")
    print(f"Location: {target_dir.relative_to(Path.cwd().parent)}")
    return True
```

---

## 📊 生成的圖表說明

### 1️⃣ **01_all_18x2_acc_round_iid_vs_label.png**
- **內容**：18 個策略在兩種分布下的精準度-輪數曲線
- **用途**：快速瞭覽各策略的收斂特性
- **格式**：寬×高（雙子圖）

### 2️⃣ **02_grouped_acc_round_iid_vs_label.png**
- **內容**：按策略類別分組的對比（Basic/Adaptive/Byzantine/DP/等）
- **用途**：比較不同類別的策略性能

### 3️⃣ **03a/03b/03c_table_*.png**
- **內容**：精準度 delta 表（IID vs Label 的差異）
- **用途**：量化各策略的非 IID 魯棒性
- **關鍵指標**：
  - **IID Acc**: 在 IID 數據上的精準度
  - **Label Acc**: 在 Label Skew 上的精準度
  - **Delta**: 差異（負值越小越好，表示非 IID 魯棒性強）

### 4️⃣ **04a/04b/04c_bar_chart_*.png**
- **內容**：策略性能條形圖（按不同指標排序）
- **用途**：快速排名各策略

### 5️⃣ **per_strategy_iid_vs_label/**
- **內容**：單個策略的詳細對比（18 個 PNG）
- **用途**：深入分析個別策略的行為

---

## 🚨 常見問題

### Q1: generate_charts 資料夾為什麼是空的？
A:
- `generator.py` 尚未執行
- 或 `PUT-DATA-THERE` 路徑列表為空（檢查 First Station 是否完成）

### Q2: tablesave 為 PNG 但格式不對怎麼辦？
A:
- matplotlib 的表格渲染可能不完美
- 改用 `pandas.to_html()` 或 `plotly` 生成互動 HTML

### Q3: move_to_stage03.py 有 robocopy 鄙誤？
A:
- Windows 路徑太長
- 確保 Windows 10/11 已啟用長路徑支持：
  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
  ```

---

## 🔗 下一步

✅ Second Station 完成 → 進入 **Final Station**

最終結果已保存在：
```
RESULT_Finally_fan_tasty/
├── {distribution}-Round-{rounds}-Epoch-{epochs}-{run_id}/
│   ├── PUT-DATA-THERE/      # 所有原始指標
│   ├── generate_charts/     # 所有對比圖表
│   └── SUMMARY.txt
```

---

## 📚 相關文檔

- [根目錄 README.md](../README.md) - 三站點流程總覽
- [First Station README](../res-drag-set-to-there----first-station/README.md) - 數據清理
- [strategies/README.md](../strategies/README.md) - 18 個策略詳解

---

**快速查詢**：三站點流程 | 生成的圖表規格 | 超參提取邏輯
