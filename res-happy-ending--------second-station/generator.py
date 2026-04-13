import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SECOND_STATION_NAME = "res-happy-ending--------second-station"

STRATEGY_ORDER = [
	"FedAvg",
	"FedAvgM",
	"FedAdagrad",
	"FedAdam",
	"FedProx",
	"FedYogi",
	"Bulyan",
	"Krum",
	"MultiKrum",
	"FedMedian",
	"FedTrimmedAvg",
	"DifferentialPrivacyClientSideAdaptiveClipping",
	"DifferentialPrivacyClientSideFixedClipping",
	"DifferentialPrivacyServerSideAdaptiveClipping",
	"DifferentialPrivacyServerSideFixedClipping",
	"FedXgbBagging",
	"FedXgbCyclic",
	"QFedAvg",
]

STRATEGY_GROUPS = {
	"Basic & Classic Strategies": ["FedAvg", "FedAvgM", "FedProx"],
	"Adaptive Optimization Strategies": ["FedAdagrad", "FedAdam", "FedYogi"],
	"Robustness & Byzantine Fault Tolerance": ["Bulyan", "Krum", "MultiKrum"],
	"Statistical Robustness Strategies": ["FedMedian", "FedTrimmedAvg"],
	"Differential Privacy Strategies": [
		"DifferentialPrivacyClientSideAdaptiveClipping",
		"DifferentialPrivacyClientSideFixedClipping",
		"DifferentialPrivacyServerSideAdaptiveClipping",
		"DifferentialPrivacyServerSideFixedClipping",
	],
	"XGBoost Specific Strategies": ["FedXgbBagging", "FedXgbCyclic"],
	"Fairness & Weighted Optimization": ["QFedAvg"],
}

DP_SHORT_NAMES = {
	"DifferentialPrivacyClientSideAdaptiveClipping": "DP-Client-Adaptive",
	"DifferentialPrivacyClientSideFixedClipping": "DP-Client-Fixed",
	"DifferentialPrivacyServerSideAdaptiveClipping": "DP-Server-Adaptive",
	"DifferentialPrivacyServerSideFixedClipping": "DP-Server-Fixed",
}


def short_name(strategy: str) -> str:
	return DP_SHORT_NAMES.get(strategy, strategy)


def parse_metadata(path: Path) -> dict[str, str]:
	info: dict[str, str] = {}
	for raw in path.read_text(encoding="utf-8").splitlines():
		line = raw.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		k, v = line.split("=", 1)
		info[k.strip()] = v.strip()
	return info


# 🔥 36 線視覺化配置：6 種色相 × 3 種線條 = 18 個子項目
# 群體 A：暖色調陣營（IID）
WARM_COLORS = [
	"#DC143C",  # 1. 正紅 (Crimson)
	"#FF8C00",  # 2. 亮橘 (Bright Orange)
	"#DAA520",  # 3. 芥末黃 (Mustard/Gold)
	"#FF1493",  # 4. 桃紅 (Magenta)
	"#8B4513",  # 5. 深棕 (Dark Brown)
	"#FF7F50",  # 6. 珊瑚紅 (Coral)
]

# 群體 B：冷色調陣營（Label）
COOL_COLORS = [
	"#000080",  # 1. 海軍藍 (Navy Blue)
	"#87CEEB",  # 2. 天藍 (Sky Blue)
	"#50C878",  # 3. 翠綠 (Emerald Green)
	"#8B00FF",  # 4. 紫羅蘭 (Violet)
	"#20B2AA",  # 5. 青綠 (Teal/Cyan)
	"#708090",  # 6. 藍灰 (Slate/Blue-Grey)
]

# 3 種線條樣式：實線、虛線、點線
LINESTYLES = ["-", "--", ":"]  # 實線、虛線、點線


def get_color_and_linestyle(strategy_idx: int, is_label: bool) -> tuple[str, str]:
	"""
	取得顏色和線條樣式。
	每 3 個策略共享一種顏色，按位置輪流使用 3 種線條樣式。
	
	組例：
	  組 0 (策略 0-2): 正紅(IID)/海軍藍(Label) × [實線, 虛線, 點線]
	  組 1 (策略 3-5): 亮橘(IID)/天藍(Label) × [實線, 虛線, 點線]
	  ...
	"""
	group_idx = strategy_idx // 3  # 色相組別 (0-5)
	style_idx = strategy_idx % 3   # 線條樣式 (0-2)
	
	if is_label:
		color = COOL_COLORS[group_idx % len(COOL_COLORS)]
	else:
		color = WARM_COLORS[group_idx % len(WARM_COLORS)]
	
	linestyle = LINESTYLES[style_idx]
	return color, linestyle


class IIDLabelComparator:
	def __init__(self, station_dir: Path):
		self.station_dir = station_dir
		self.input_root = station_dir / "PUT-DATA-THERE"
		self.output_root = station_dir / "generate_charts"
		self.output_root.mkdir(parents=True, exist_ok=True)
		self.per_strategy_dir = self.output_root / "per_strategy_iid_vs_label"
		self.per_strategy_dir.mkdir(parents=True, exist_ok=True)

		self.strategy_index = {s: i for i, s in enumerate(STRATEGY_ORDER)}
		
		# 新的熱冷色調系統：6 色相 × 3 線條樣式
		self.iid_colors = {}
		self.iid_styles = {}
		self.label_colors = {}
		self.label_styles = {}
		
		for strategy in STRATEGY_ORDER:
			idx = self.strategy_index[strategy]
			iid_color, iid_style = get_color_and_linestyle(idx, is_label=False)
			label_color, label_style = get_color_and_linestyle(idx, is_label=True)
			
			self.iid_colors[strategy] = iid_color
			self.iid_styles[strategy] = iid_style
			self.label_colors[strategy] = label_color
			self.label_styles[strategy] = label_style

		# data[distribution][strategy] = {"df": DataFrame, "run_id": str, "elapsed_seconds": float}
		self.data: dict[str, dict[str, dict[str, object]]] = {"iid": {}, "label": {}}

	def load(self) -> None:
		if not self.input_root.exists():
			raise FileNotFoundError(f"Input root not found: {self.input_root}")

		import json
		batch_dirs = [d for d in sorted(self.input_root.iterdir()) if d.is_dir()]
		for batch_dir in batch_dirs:
			run_root = batch_dir / "PUT-DATA-THERE"
			if not run_root.exists():
				continue

			for run_dir in sorted(run_root.iterdir()):
				if not run_dir.is_dir():
					continue
				metadata_path = run_dir / "metadata.txt"
				metrics_csv_path = run_dir / "run_metrics.csv"
				metrics_json_path = run_dir / "run_metrics.json"
				
				# Check metadata exists
				if not metadata_path.exists():
					continue

				meta = parse_metadata(metadata_path)
				strategy = meta.get("strategy")
				distribution = meta.get("data_distribution", "").lower()
				run_id = meta.get("run_id", run_dir.name)
				elapsed = float(meta.get("elapsed_seconds", "nan"))

				if strategy not in STRATEGY_ORDER:
					continue
				if distribution not in {"iid", "label"}:
					continue

				# Try to load from CSV first, then JSON
				# Use Windows long path prefix for paths >= 260 chars to bypass MAX_PATH limit
				df = None
				csv_abs = metrics_csv_path.absolute()
				json_abs = metrics_json_path.absolute()
				
				# Apply Windows long path prefix when path is >= 260 chars
				import sys
				if sys.platform == "win32" and len(str(csv_abs)) >= 260:
					csv_str = f"\\\\?\\{str(csv_abs)}"
					json_str = f"\\\\?\\{str(json_abs)}"
				else:
					csv_str = str(csv_abs)
					json_str = str(json_abs)
				
				try:
					df = pd.read_csv(csv_str)
				except (FileNotFoundError, OSError):
					try:
						with open(json_str, "r") as f:
							json_data = json.load(f)
							if isinstance(json_data, dict) and "rounds" in json_data:
								json_data = json_data["rounds"]
							df = pd.DataFrame(json_data)
					except (FileNotFoundError, OSError, json.JSONDecodeError):
						continue
			
				if df is None or not {"round", "accuracy"}.issubset(df.columns):
					continue

				current = self.data[distribution].get(strategy)
				if current is None or str(run_id) > str(current.get("run_id", "")):
					self.data[distribution][strategy] = {
						"df": df,
						"run_id": run_id,
						"elapsed_seconds": elapsed,
					}

	def _has_both(self, strategy: str) -> bool:
		return strategy in self.data["iid"] and strategy in self.data["label"]

	def plot_all_18x2(self) -> None:
		fig, ax = plt.subplots(figsize=(18, 11))

		for strategy in STRATEGY_ORDER:
			if strategy in self.data["iid"]:
				df_iid = self.data["iid"][strategy]["df"]
				ax.plot(
					df_iid["round"],
					df_iid["accuracy"],
					color=self.iid_colors[strategy],
					linestyle=self.iid_styles[strategy],
					marker="o",
					markersize=3,
					linewidth=1.3,
					alpha=0.85,
					label=f"IID | {short_name(strategy)}",
				)
			if strategy in self.data["label"]:
				df_label = self.data["label"][strategy]["df"]
				ax.plot(
					df_label["round"],
					df_label["accuracy"],
					color=self.label_colors[strategy],
					linestyle=self.label_styles[strategy],
					marker="s",
					markersize=3,
					linewidth=1.3,
					alpha=0.85,
					label=f"Label | {short_name(strategy)}",
				)

		ax.set_title(
			"Comprehensive Comparison of 18 Strategies: IID vs Label Distribution\nAccuracy over Training Rounds (6 Colors × 3 Line Styles)",
			fontsize=14,
			fontweight="bold",
		)
		ax.set_xlabel("Training Round", fontsize=11, fontweight="bold")
		ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
		ax.grid(True, alpha=0.25)
		# 圖例分成左右兩欄
		ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=2, frameon=True)
		plt.tight_layout()
		plt.savefig(self.output_root / "01_all_18x2_acc_round_iid_vs_label.png", dpi=160, bbox_inches="tight")
		plt.close()

	def plot_per_strategy(self) -> None:
		for strategy in STRATEGY_ORDER:
			if not self._has_both(strategy):
				continue

			fig, ax = plt.subplots(figsize=(9, 5.5))
			df_iid = self.data["iid"][strategy]["df"]
			df_label = self.data["label"][strategy]["df"]

			ax.plot(
				df_iid["round"],
				df_iid["accuracy"],
				color=self.iid_colors[strategy],
				linestyle=self.iid_styles[strategy],
				marker="o",
				markersize=4,
				linewidth=2.2,
				alpha=0.85,
				label="IID (Warm Tone)",
			)
			ax.plot(
				df_label["round"],
				df_label["accuracy"],
				color=self.label_colors[strategy],
				linestyle=self.label_styles[strategy],
				marker="s",
				markersize=4,
				linewidth=2.2,
				alpha=0.85,
				label="Label (Cool Tone)",
			)

			ax.set_title(f"{short_name(strategy)}: IID vs Label Distribution", fontsize=12, fontweight="bold")
			ax.set_xlabel("Training Round", fontsize=10)
			ax.set_ylabel("Accuracy", fontsize=10)
			ax.grid(True, alpha=0.25)
			ax.legend(loc="best", fontsize=9)
			plt.tight_layout()
			plt.savefig(self.per_strategy_dir / f"{strategy}_iid_vs_label.png", dpi=160, bbox_inches="tight")
			plt.close()

	def plot_grouped(self) -> None:
		fig, axes = plt.subplots(4, 2, figsize=(17, 17))
		axes = axes.flatten()
		group_names = list(STRATEGY_GROUPS.keys())

		for idx, group_name in enumerate(group_names):
			ax = axes[idx]
			for strategy in STRATEGY_GROUPS[group_name]:
				if strategy in self.data["iid"]:
					df_iid = self.data["iid"][strategy]["df"]
					ax.plot(
						df_iid["round"],
						df_iid["accuracy"],
						color=self.iid_colors[strategy],
						linestyle=self.iid_styles[strategy],
						marker="o",
						markersize=3,
						linewidth=1.6,
						alpha=0.85,
						label=f"IID | {short_name(strategy)}",
					)
				if strategy in self.data["label"]:
					df_label = self.data["label"][strategy]["df"]
					ax.plot(
						df_label["round"],
						df_label["accuracy"],
						color=self.label_colors[strategy],
						linestyle=self.label_styles[strategy],
						marker="s",
						markersize=3,
						linewidth=1.6,
						alpha=0.85,
						label=f"Label | {short_name(strategy)}",
					)

			ax.set_title(group_name, fontsize=11, fontweight="bold")
			ax.set_xlabel("Round", fontsize=9)
			ax.set_ylabel("Accuracy", fontsize=9)
			ax.grid(True, alpha=0.25)
			ax.legend(fontsize=7, loc="best")

		for extra in range(len(group_names), len(axes)):
			axes[extra].axis("off")

		fig.suptitle("Grouped IID vs Label Comparison (6 Colors × 3 Line Styles)", fontsize=13, fontweight="bold")
		plt.tight_layout()
		plt.savefig(self.output_root / "02_grouped_acc_round_iid_vs_label.png", dpi=160, bbox_inches="tight")
		plt.close()

	def build_delta_table(self) -> pd.DataFrame:
		rows: list[dict[str, object]] = []
		group_lookup = {s: g for g, ss in STRATEGY_GROUPS.items() for s in ss}

		for strategy in STRATEGY_ORDER:
			iid_final = np.nan
			label_final = np.nan
			if strategy in self.data["iid"]:
				iid_final = float(self.data["iid"][strategy]["df"]["accuracy"].iloc[-1]) * 100.0
			if strategy in self.data["label"]:
				label_final = float(self.data["label"][strategy]["df"]["accuracy"].iloc[-1]) * 100.0
			delta = label_final - iid_final if not (np.isnan(iid_final) or np.isnan(label_final)) else np.nan

			rows.append(
				{
					"group": group_lookup.get(strategy, "Unknown"),
					"strategy": strategy,
					"iid_final_acc": iid_final,
					"label_final_acc": label_final,
					"delta_label_minus_iid": delta,
				}
			)

		return pd.DataFrame(rows)

	def build_best_vs_final_delta_table(self) -> pd.DataFrame:
		"""Build table with best history vs final round accuracy delta for both IID and Label."""
		rows: list[dict[str, object]] = []
		group_lookup = {s: g for g, ss in STRATEGY_GROUPS.items() for s in ss}

		for strategy in STRATEGY_ORDER:
			iid_best = np.nan
			iid_final = np.nan
			iid_delta = np.nan
			label_best = np.nan
			label_final = np.nan
			label_delta = np.nan
			iid_vs_label_delta = np.nan
			
			if strategy in self.data["iid"]:
				df_iid = self.data["iid"][strategy]["df"]
				iid_best = float(df_iid["accuracy"].max()) * 100.0
				iid_final = float(df_iid["accuracy"].iloc[-1]) * 100.0
				iid_delta = iid_best - iid_final  # best minus final
			
			if strategy in self.data["label"]:
				df_label = self.data["label"][strategy]["df"]
				label_best = float(df_label["accuracy"].max()) * 100.0
				label_final = float(df_label["accuracy"].iloc[-1]) * 100.0
				label_delta = label_best - label_final  # best minus final
			
			if not (np.isnan(iid_final) or np.isnan(label_final)):
				iid_vs_label_delta = label_final - iid_final

			rows.append(
				{
					"group": group_lookup.get(strategy, "Unknown"),
					"strategy": strategy,
					"iid_best_acc": iid_best,
					"iid_final_acc": iid_final,
					"iid_best_vs_final_delta": iid_delta,
					"label_best_acc": label_best,
					"label_final_acc": label_final,
					"label_best_vs_final_delta": label_delta,
					"iid_vs_label_final_delta": iid_vs_label_delta,
				}
			)

		return pd.DataFrame(rows)

	def _render_table_png(self, df: pd.DataFrame, output_name: str, title: str) -> None:
		render_df = df.copy()
		render_df["iid_final_acc"] = render_df["iid_final_acc"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")
		render_df["label_final_acc"] = render_df["label_final_acc"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")
		render_df["delta_label_minus_iid"] = render_df["delta_label_minus_iid"].map(
			lambda x: "-" if pd.isna(x) else f"{x:+.2f}%"
		)

		fig, ax = plt.subplots(figsize=(16, 10))
		ax.axis("off")
		table_data = render_df[["group", "strategy", "iid_final_acc", "label_final_acc", "delta_label_minus_iid"]].values.tolist()
		col_labels = ["Group", "Strategy", "IID Final", "Label Final", "Delta (Label - IID)"]

		raw_delta = df["delta_label_minus_iid"].copy()
		top3_idx = set(raw_delta.dropna().sort_values(ascending=False).head(3).index.tolist())

		cell_colors = [["#ffffff"] * len(col_labels) for _ in table_data]
		for i, row_idx in enumerate(df.index.tolist()):
			if row_idx in top3_idx:
				cell_colors[i][4] = "#ffcccc"

		table = ax.table(
			cellText=table_data,
			colLabels=col_labels,
			cellLoc="center",
			loc="center",
			cellColours=cell_colors,
			colColours=["#e6e6e6"] * len(col_labels),
		)
		table.auto_set_font_size(False)
		table.set_fontsize(8)
		table.scale(1, 2.2)

		for c in range(len(col_labels)):
			table[(0, c)].set_facecolor("#8a8a8a")
			table[(0, c)].set_text_props(weight="bold", color="white")

		fig.suptitle(title, fontsize=13, fontweight="bold")
		fig.text(
			0.5,
			0.04,
			"Red shading: Top-3 largest positive Delta (Label - IID)",
			ha="center",
			fontsize=10,
			style="italic",
		)
		plt.tight_layout()
		plt.savefig(self.output_root / output_name, dpi=160, bbox_inches="tight")
		plt.close()

	def _render_best_vs_final_table_png(self, df: pd.DataFrame, output_name: str, title: str) -> None:
		"""Render table with best history vs final round accuracy delta."""
		render_df = df.copy()
		
		# Format all numeric columns - keep original for highlighting logic
		formatted_df = render_df.copy()
		for col in ["iid_best_acc", "iid_final_acc", "iid_best_vs_final_delta", 
					"label_best_acc", "label_final_acc", "label_best_vs_final_delta", 
					"iid_vs_label_final_delta"]:
			if col in formatted_df.columns:
				if "delta" in col:
					formatted_df[col] = formatted_df[col].map(lambda x: "-" if pd.isna(x) else f"{x:+.2f}%")
				else:
					formatted_df[col] = formatted_df[col].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")
		
		fig, ax = plt.subplots(figsize=(20, 12))
		ax.axis("off")
		
		table_data = formatted_df[[
			"group", "strategy", 
			"iid_best_acc", "iid_final_acc", "iid_best_vs_final_delta",
			"label_best_acc", "label_final_acc", "label_best_vs_final_delta",
			"iid_vs_label_final_delta"
		]].values.tolist()
		
		col_labels = [
			"Group", "Strategy",
			"IID Best", "IID Final", "IID Δ(B-F)",
			"Label Best", "Label Final", "Label Δ(B-F)", 
			"Δ(Label-IID)"
		]
		
		# Find top deltas for highlighting
		iid_delta = df["iid_best_vs_final_delta"].copy()
		label_delta = df["label_best_vs_final_delta"].copy()
		iid_final_delta = df["iid_vs_label_final_delta"].copy()
		
		top3_iid_delta_idx = set(iid_delta.dropna().nlargest(3).index.tolist()) if len(iid_delta.dropna()) > 0 else set()
		top3_label_delta_idx = set(label_delta.dropna().nlargest(3).index.tolist()) if len(label_delta.dropna()) > 0 else set()
		top3_iid_vs_label_idx = set(iid_final_delta.dropna().nlargest(3).index.tolist()) if len(iid_final_delta.dropna()) > 0 else set()
		
		cell_colors = [["#ffffff"] * len(col_labels) for _ in table_data]
		for i, row_idx in enumerate(df.index.tolist()):
			if row_idx in top3_iid_delta_idx:
				cell_colors[i][4] = "#ffe6e6"  # Light red for IID delta
			if row_idx in top3_label_delta_idx:
				cell_colors[i][7] = "#ffe6e6"  # Light red for Label delta
			if row_idx in top3_iid_vs_label_idx:
				cell_colors[i][8] = "#ffcccc"  # Darker red for IID vs Label
		
		table = ax.table(
			cellText=table_data,
			colLabels=col_labels,
			cellLoc="center",
			loc="center",
			cellColours=cell_colors,
			colColours=["#e6e6e6"] * len(col_labels),
		)
		table.auto_set_font_size(False)
		table.set_fontsize(7.5)
		table.scale(1, 1.8)
		
		for c in range(len(col_labels)):
			table[(0, c)].set_facecolor("#8a8a8a")
			table[(0, c)].set_text_props(weight="bold", color="white")
		
		fig.suptitle(title, fontsize=13, fontweight="bold")
		fig.text(
			0.5,
			0.02,
			"Light Red: Top-3 Largest Best-vs-Final Delta | Dark Red: Top-3 Largest IID-vs-Label Final Delta",
			ha="center",
			fontsize=9,
			style="italic",
		)
		plt.tight_layout()
		plt.savefig(self.output_root / output_name, dpi=160, bbox_inches="tight")
		plt.close()

	def plot_bar_chart(self) -> None:
		"""Plot bar chart of IID vs Label final accuracy, sorted by delta."""
		df = self.build_delta_table()
		df_sorted = df.sort_values("delta_label_minus_iid", ascending=False)
		top4_delta = set(df["delta_label_minus_iid"].nlargest(4).index.tolist())
		
		fig, ax = plt.subplots(figsize=(18, 8))
		
		x_pos = np.arange(len(df_sorted))
		bar_width = 0.35
		
		# 準備資料
		strategies = df_sorted["strategy"].values
		iid_accs = df_sorted["iid_final_acc"].values
		label_accs = df_sorted["label_final_acc"].values
		deltas = df_sorted["delta_label_minus_iid"].values
		
		# 繪製柱狀圖
		bars1 = ax.bar(
			x_pos - bar_width/2,
			iid_accs,
			bar_width,
			label="IID (Warm Tone)",
			color="#FFB347",  # 暖色（珊瑚橙）
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		
		bars2 = ax.bar(
			x_pos + bar_width/2,
			label_accs,
			bar_width,
			label="Label (Cool Tone)",
			color="#87CEEB",  # 冷色（天藍）
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		
		# 在柱子上方添加 delta 值標籤（僅前四名標紅）
		for i, (idx, delta, x) in enumerate(zip(df_sorted.index, deltas, x_pos)):
			if not np.isnan(delta):
				label_text = f"{delta:+.1f}%" if abs(delta) >= 0.1 else ""
				color = "#cc0000" if idx in top4_delta else "#333333"
				ax.text(x, max(iid_accs[i], label_accs[i]) + 2, label_text, 
					   ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)
		
		# 設定 X 軸標籤
		short_names = [short_name(s) for s in strategies]
		ax.set_xticks(x_pos)
		ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
		
		ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
		ax.set_title(
			"IID vs Label Final Accuracy Comparison (Sorted by Delta)\n36 Strategies Bar Chart [Red Delta = Top-4 by Delta]",
			fontsize=14,
			fontweight="bold",
			pad=20
		)
		
		ax.set_ylim(0, 110)
		ax.grid(True, axis="y", alpha=0.3, linestyle="--")
		ax.legend(fontsize=11, loc="upper right")
		
		plt.tight_layout()
		plt.savefig(self.output_root / "04a_bar_chart_sorted_by_delta.png", dpi=160, bbox_inches="tight")
		plt.close()

	def plot_bar_chart_sorted_by_iid(self) -> None:
		"""Plot bar chart of IID vs Label final accuracy, sorted by IID accuracy.
		Show delta labels and highlight top-4 best delta labels in red."""
		df = self.build_delta_table()
		df_sorted = df.sort_values("iid_final_acc", ascending=False)

		# 找出 delta 最高的前 4 個策略（與 04a 一致）
		top4_delta = set(df["delta_label_minus_iid"].nlargest(4).index.tolist())
		
		fig, ax = plt.subplots(figsize=(18, 8))
		
		x_pos = np.arange(len(df_sorted))
		bar_width = 0.35
		
		# 準備資料
		strategies = df_sorted["strategy"].values
		iid_accs = df_sorted["iid_final_acc"].values
		label_accs = df_sorted["label_final_acc"].values
		deltas = df_sorted["delta_label_minus_iid"].values

		# 保持柱狀顏色一致，僅以 delta 文字標紅前四名
		ax.bar(
			x_pos - bar_width/2,
			iid_accs,
			bar_width,
			label="IID (Warm Tone)",
			color="#FFB347",
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		ax.bar(
			x_pos + bar_width/2,
			label_accs,
			bar_width,
			label="Label (Cool Tone)",
			color="#87CEEB",
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		
		# 在柱子上方添加數值標籤
		for i, (iid_acc, label_acc, x) in enumerate(zip(iid_accs, label_accs, x_pos)):
			if not np.isnan(iid_acc):
				ax.text(x - bar_width/2, iid_acc + 1, f"{iid_acc:.0f}", 
					   ha="center", va="bottom", fontsize=7, color="#333333")
			if not np.isnan(label_acc):
				ax.text(x + bar_width/2, label_acc + 1, f"{label_acc:.0f}", 
					   ha="center", va="bottom", fontsize=7, color="#333333")

		# 加上 delta 文字，僅前四名標紅
		for i, (idx, delta, x) in enumerate(zip(df_sorted.index, deltas, x_pos)):
			if not np.isnan(delta):
				label_color = "#cc0000" if idx in top4_delta else "#333333"
				ax.text(
					x,
					max(iid_accs[i], label_accs[i]) + 3,
					f"{delta:+.1f}%",
					ha="center",
					va="bottom",
					fontsize=8,
					fontweight="bold",
					color=label_color,
				)
		
		# 設定 X 軸標籤
		short_names = [short_name(s) for s in strategies]
		ax.set_xticks(x_pos)
		ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
		
		ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
		ax.set_title(
			"IID vs Label Final Accuracy Comparison (Sorted by IID Accuracy)\n36 Strategies Bar Chart [Red Delta = Top-4 by Delta]",
			fontsize=14,
			fontweight="bold",
			pad=20
		)
		
		ax.set_ylim(0, 110)
		ax.grid(True, axis="y", alpha=0.3, linestyle="--")
		ax.legend(fontsize=11, loc="upper right")
		
		plt.tight_layout()
		plt.savefig(self.output_root / "04b_bar_chart_sorted_by_iid.png", dpi=160, bbox_inches="tight")
		plt.close()

	def plot_bar_chart_sorted_by_label(self) -> None:
		"""Plot bar chart of IID vs Label final accuracy, sorted by Label accuracy.
		Show delta labels and highlight top-4 best delta labels in red."""
		df = self.build_delta_table()
		df_sorted = df.sort_values("label_final_acc", ascending=False)

		# 找出 delta 最高的前 4 個策略（與 04a 一致）
		top4_delta = set(df["delta_label_minus_iid"].nlargest(4).index.tolist())
		
		fig, ax = plt.subplots(figsize=(18, 8))
		
		x_pos = np.arange(len(df_sorted))
		bar_width = 0.35
		
		# 準備資料
		strategies = df_sorted["strategy"].values
		iid_accs = df_sorted["iid_final_acc"].values
		label_accs = df_sorted["label_final_acc"].values
		deltas = df_sorted["delta_label_minus_iid"].values

		# 保持柱狀顏色一致，僅以 delta 文字標紅前四名
		ax.bar(
			x_pos - bar_width/2,
			iid_accs,
			bar_width,
			label="IID (Warm Tone)",
			color="#FFB347",
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		ax.bar(
			x_pos + bar_width/2,
			label_accs,
			bar_width,
			label="Label (Cool Tone)",
			color="#87CEEB",
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		
		# 在柱子上方添加數值標籤
		for i, (iid_acc, label_acc, x) in enumerate(zip(iid_accs, label_accs, x_pos)):
			if not np.isnan(iid_acc):
				ax.text(x - bar_width/2, iid_acc + 1, f"{iid_acc:.0f}", 
					   ha="center", va="bottom", fontsize=7, color="#333333")
			if not np.isnan(label_acc):
				ax.text(x + bar_width/2, label_acc + 1, f"{label_acc:.0f}", 
					   ha="center", va="bottom", fontsize=7, color="#333333")

		# 加上 delta 文字，僅前四名標紅
		for i, (idx, delta, x) in enumerate(zip(df_sorted.index, deltas, x_pos)):
			if not np.isnan(delta):
				label_color = "#cc0000" if idx in top4_delta else "#333333"
				ax.text(
					x,
					max(iid_accs[i], label_accs[i]) + 3,
					f"{delta:+.1f}%",
					ha="center",
					va="bottom",
					fontsize=8,
					fontweight="bold",
					color=label_color,
				)
		
		# 設定 X 軸標籤
		short_names = [short_name(s) for s in strategies]
		ax.set_xticks(x_pos)
		ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
		
		ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
		ax.set_title(
			"IID vs Label Final Accuracy Comparison (Sorted by Label Accuracy)\n36 Strategies Bar Chart [Red Delta = Top-4 by Delta]",
			fontsize=14,
			fontweight="bold",
			pad=20
		)
		
		ax.set_ylim(0, 110)
		ax.grid(True, axis="y", alpha=0.3, linestyle="--")
		ax.legend(fontsize=11, loc="upper right")
		
		plt.tight_layout()
		plt.savefig(self.output_root / "04c_bar_chart_sorted_by_label.png", dpi=160, bbox_inches="tight")
		plt.close()

	def make_table_grouped_by_category(self) -> None:
		"""Generate 03c table: Grouped by strategy category, sorted by group then delta within group.
		Highlight the best delta (highest/least negative) in each group."""
		df = self.build_delta_table()
		group_order = {g: i for i, g in enumerate(STRATEGY_GROUPS.keys())}
		
		# 按組排序，組內按 delta 降序排列（最高的在前）
		df["_g"] = df["group"].map(lambda g: group_order.get(g, 999))
		df_grouped = df.sort_values(["_g", "delta_label_minus_iid"], ascending=[True, False]).drop(columns=["_g"])
		
		# 找出每個組內 delta 最高（最異常/最少負）的策略
		best_delta_per_group = {}
		for group in STRATEGY_GROUPS.keys():
			group_rows = df_grouped[df_grouped["group"] == group]
			if not group_rows.empty:
				best_idx = group_rows["delta_label_minus_iid"].idxmax()
				best_delta_per_group[best_idx] = group
		
		# 渲染表格（用群組標題列 + 區塊底色來分隔，不用粗線）
		render_df = df_grouped.copy()
		render_df["iid_final_acc"] = render_df["iid_final_acc"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")
		render_df["label_final_acc"] = render_df["label_final_acc"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}%")
		render_df["delta_label_minus_iid"] = render_df["delta_label_minus_iid"].map(
			lambda x: "-" if pd.isna(x) else f"{x:+.2f}%"
		)

		table_data: list[list[str]] = []
		row_meta: list[tuple[str, str | int | None]] = []

		for g_idx, group in enumerate(STRATEGY_GROUPS.keys()):
			group_rows = render_df[render_df["group"] == group]
			if group_rows.empty:
				continue

			# 群組標題列
			table_data.append([f"[{group}]", "", "", "", ""])
			row_meta.append(("group_header", group))

			for _, row in group_rows.iterrows():
				table_data.append(
					[
						str(row["group"]),
						str(row["strategy"]),
						str(row["iid_final_acc"]),
						str(row["label_final_acc"]),
						str(row["delta_label_minus_iid"]),
					]
				)
				row_meta.append(("data", int(row.name)))

			# 群組空白分隔列（最後一組不加）
			if g_idx < len(STRATEGY_GROUPS) - 1:
				table_data.append(["", "", "", "", ""])
				row_meta.append(("spacer", None))

		fig, ax = plt.subplots(figsize=(16, 13))
		ax.axis("off")
		col_labels = ["Group", "Strategy", "IID Final", "Label Final", "Delta (Label - IID)"]

		cell_colors = [["#ffffff"] * len(col_labels) for _ in table_data]
		table = ax.table(
			cellText=table_data,
			colLabels=col_labels,
			cellLoc="center",
			loc="center",
			cellColours=cell_colors,
			colColours=["#e6e6e6"] * len(col_labels),
		)
		table.auto_set_font_size(False)
		table.set_fontsize(8)
		table.scale(1, 1.9)

		# 依列型態美化：群組標題列/資料列/空白列
		for r, meta in enumerate(row_meta, start=1):
			kind, payload = meta
			if kind == "group_header":
				for c in range(len(col_labels)):
					cell = table[(r, c)]
					cell.set_facecolor("#dbe7f3")
					cell.set_linewidth(0.9)
					if c == 0:
						cell.set_text_props(weight="bold", color="#1f3b57")
					else:
						cell.get_text().set_text("")
			elif kind == "spacer":
				for c in range(len(col_labels)):
					cell = table[(r, c)]
					cell.set_facecolor("#ffffff")
					cell.set_linewidth(0.0)
					cell.get_text().set_text("")
			else:
				row_idx = int(payload)
				group_name = str(df_grouped.loc[row_idx, "group"])
				band = "#fbfdff" if group_order.get(group_name, 0) % 2 == 0 else "#f9fcf6"
				for c in range(len(col_labels)):
					cell = table[(r, c)]
					cell.set_facecolor(band)
					cell.set_linewidth(0.45)
				if row_idx in best_delta_per_group:
					table[(r, 4)].set_facecolor("#ffcccc")
		
		for c in range(len(col_labels)):
			table[(0, c)].set_facecolor("#8a8a8a")
			table[(0, c)].set_text_props(weight="bold", color="white")
		
		fig.suptitle("Final Accuracy Delta by Strategy Group (Sorted by Delta within Group)", fontsize=13, fontweight="bold")
		fig.text(
			0.5,
			0.04,
			"Red shading: Best (highest/least negative) Delta within each group",
			ha="center",
			fontsize=10,
			style="italic",
		)
		plt.tight_layout()
		plt.savefig(self.output_root / "03c_table_final_acc_delta_grouped_by_category.png", dpi=160, bbox_inches="tight")
		plt.close()
		
		df_grouped.to_csv(self.output_root / "03c_final_acc_delta_grouped_by_category.csv", index=False)

	def make_tables(self) -> None:
		df = self.build_delta_table()
		group_order = {g: i for i, g in enumerate(STRATEGY_GROUPS.keys())}
		df_grouped = df.copy()
		df_grouped["_g"] = df_grouped["group"].map(lambda g: group_order.get(g, 999))
		df_grouped["_s"] = df_grouped["strategy"].map(lambda s: self.strategy_index.get(s, 999))
		df_grouped = df_grouped.sort_values(["_g", "_s"]).drop(columns=["_g", "_s"])

		df_sorted = df.sort_values("delta_label_minus_iid", ascending=False)

		df_grouped.to_csv(self.output_root / "03a_final_acc_delta_by_group.csv", index=False)
		df_sorted.to_csv(self.output_root / "03b_final_acc_delta_sorted.csv", index=False)

		self._render_table_png(
			df_grouped,
			"03a_table_final_acc_delta_grouped.png",
			"Final Accuracy Delta by Strategy Group (IID vs Label)",
		)
		self._render_table_png(
			df_sorted,
			"03b_table_final_acc_delta_sorted.png",
			"Final Accuracy Delta Sorted (Label - IID)",
		)

	def make_best_vs_final_tables(self) -> None:
		"""Generate tables showing best history vs final round accuracy delta."""
		df = self.build_best_vs_final_delta_table()
		group_order = {g: i for i, g in enumerate(STRATEGY_GROUPS.keys())}
		
		# Grouped by category
		df_grouped = df.copy()
		df_grouped["_g"] = df_grouped["group"].map(lambda g: group_order.get(g, 999))
		df_grouped["_s"] = df_grouped["strategy"].map(lambda s: self.strategy_index.get(s, 999))
		df_grouped = df_grouped.sort_values(["_g", "_s"]).drop(columns=["_g", "_s"])
		
		# Sorted by best-vs-final delta (IID)
		df_sorted_iid = df.sort_values("iid_best_vs_final_delta", ascending=False)
		
		# Sorted by best-vs-final delta (Label)
		df_sorted_label = df.sort_values("label_best_vs_final_delta", ascending=False)
		
		# Save to CSV
		df_grouped.to_csv(self.output_root / "05a_best_vs_final_delta_by_group.csv", index=False)
		df_sorted_iid.to_csv(self.output_root / "05b_best_vs_final_delta_sorted_iid.csv", index=False)
		df_sorted_label.to_csv(self.output_root / "05c_best_vs_final_delta_sorted_label.csv", index=False)
		
		# Render tables
		self._render_best_vs_final_table_png(
			df_grouped,
			"05a_table_best_vs_final_delta_grouped.png",
			"Best History vs Final Round Accuracy: Grouped by Category",
		)
		self._render_best_vs_final_table_png(
			df_sorted_iid,
			"05b_table_best_vs_final_delta_sorted_iid.png",
			"Best History vs Final Round Accuracy: Sorted by IID Delta",
		)
		self._render_best_vs_final_table_png(
			df_sorted_label,
			"05c_table_best_vs_final_delta_sorted_label.png",
			"Best History vs Final Round Accuracy: Sorted by Label Delta",
		)

	def plot_best_vs_final_bar_chart_sorted_by_delta(self) -> None:
		"""Plot bar chart of best vs final accuracy delta, sorted by combined delta."""
		df = self.build_best_vs_final_delta_table()
		# Calculate average delta across IID and Label (only where both exist)
		df["avg_delta"] = df.apply(
			lambda row: (row["iid_best_vs_final_delta"] + row["label_best_vs_final_delta"]) / 2 
			if not (np.isnan(row["iid_best_vs_final_delta"]) or np.isnan(row["label_best_vs_final_delta"])) 
			else np.nanmax([row["iid_best_vs_final_delta"], row["label_best_vs_final_delta"]]),
			axis=1
		)
		df_sorted = df.sort_values("avg_delta", ascending=False)
		
		fig, ax = plt.subplots(figsize=(18, 7))
		x_pos = np.arange(len(df_sorted))
		bar_width = 0.35
		
		strategies = df_sorted["strategy"].values
		iid_deltas = df_sorted["iid_best_vs_final_delta"].values
		label_deltas = df_sorted["label_best_vs_final_delta"].values
		
		# Replace NaN with 0 for plotting (but we'll handle them in labels)
		iid_deltas_plot = np.nan_to_num(iid_deltas, nan=0.0)
		label_deltas_plot = np.nan_to_num(label_deltas, nan=0.0)
		
		bars1 = ax.bar(
			x_pos - bar_width/2,
			iid_deltas_plot,
			bar_width,
			label="IID (Best - Final)",
			color="#FF9999",
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		bars2 = ax.bar(
			x_pos + bar_width/2,
			label_deltas_plot,
			bar_width,
			label="Label (Best - Final)",
			color="#99CCFF",
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		
		# Add value labels on bars (only if not NaN)
		for i, (iid_d, label_d, x) in enumerate(zip(iid_deltas, label_deltas, x_pos)):
			if not np.isnan(iid_d) and iid_d > 0.1:
				ax.text(x - bar_width/2, iid_d + 0.2, f"{iid_d:.1f}", 
					   ha="center", va="bottom", fontsize=7, color="#333333")
			if not np.isnan(label_d) and label_d > 0.1:
				ax.text(x + bar_width/2, label_d + 0.2, f"{label_d:.1f}", 
					   ha="center", va="bottom", fontsize=7, color="#333333")
		
		short_names = [short_name(s) for s in strategies]
		ax.set_xticks(x_pos)
		ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
		
		ax.set_ylabel("Accuracy Loss (%)", fontsize=11, fontweight="bold")
		ax.set_title(
			"Best vs Final Round Accuracy Delta (Sorted by Average Delta)\n[Higher = More Loss from Peak to Final]",
			fontsize=13,
			fontweight="bold",
			pad=15
		)
		
		ax.grid(True, axis="y", alpha=0.3, linestyle="--")
		ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
		ax.set_ylim(bottom=0, top=max(np.nanmax(iid_deltas), np.nanmax(label_deltas)) * 1.15 if len(iid_deltas) > 0 else 50)
		
		plt.tight_layout()
		plt.savefig(self.output_root / "06a_bar_chart_best_vs_final_sorted_by_delta.png", dpi=160, bbox_inches="tight")
		plt.close()

	def plot_best_vs_final_bar_chart_by_distribution(self, distribution: str) -> None:
		"""Plot bar chart of best vs final accuracy delta, sorted by specific distribution."""
		df = self.build_best_vs_final_delta_table()
		
		if distribution.lower() == "iid":
			df_sorted = df.sort_values("iid_best_vs_final_delta", ascending=False, na_position="last")
			delta_col = "iid_best_vs_final_delta"
			filename = "06b_bar_chart_best_vs_final_sorted_by_iid.png"
			title = "Best vs Final Round Accuracy Delta (IID Distribution)"
			color1 = "#FF9999"
			color2 = "#99CCFF"
		else:
			df_sorted = df.sort_values("label_best_vs_final_delta", ascending=False, na_position="last")
			delta_col = "label_best_vs_final_delta"
			filename = "06c_bar_chart_best_vs_final_sorted_by_label.png"
			title = "Best vs Final Round Accuracy Delta (Label Distribution)"
			color1 = "#FF9999"
			color2 = "#99CCFF"
		
		fig, ax = plt.subplots(figsize=(18, 7))
		x_pos = np.arange(len(df_sorted))
		bar_width = 0.35
		
		strategies = df_sorted["strategy"].values
		iid_deltas = df_sorted["iid_best_vs_final_delta"].values
		label_deltas = df_sorted["label_best_vs_final_delta"].values
		
		# Replace NaN with 0 for plotting
		iid_deltas_plot = np.nan_to_num(iid_deltas, nan=0.0)
		label_deltas_plot = np.nan_to_num(label_deltas, nan=0.0)
		
		bars1 = ax.bar(
			x_pos - bar_width/2,
			iid_deltas_plot,
			bar_width,
			label="IID (Best - Final)",
			color=color1,
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		bars2 = ax.bar(
			x_pos + bar_width/2,
			label_deltas_plot,
			bar_width,
			label="Label (Best - Final)",
			color=color2,
			alpha=0.85,
			edgecolor="black",
			linewidth=0.5,
		)
		
		# Add value labels on bars (only if not NaN)
		for i, (iid_d, label_d, x) in enumerate(zip(iid_deltas, label_deltas, x_pos)):
			if not np.isnan(iid_d) and iid_d > 0.1:
				ax.text(x - bar_width/2, iid_d + 0.2, f"{iid_d:.1f}", 
					   ha="center", va="bottom", fontsize=7, color="#333333")
			if not np.isnan(label_d) and label_d > 0.1:
				ax.text(x + bar_width/2, label_d + 0.2, f"{label_d:.1f}", 
					   ha="center", va="bottom", fontsize=7, color="#333333")
		
		short_names = [short_name(s) for s in strategies]
		ax.set_xticks(x_pos)
		ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
		
		ax.set_ylabel("Accuracy Loss (%)", fontsize=11, fontweight="bold")
		ax.set_title(
			f"{title}\n[Higher = More Loss from Peak to Final]",
			fontsize=13,
			fontweight="bold",
			pad=15
		)
		
		ax.grid(True, axis="y", alpha=0.3, linestyle="--")
		ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
		max_delta = np.nanmax(np.concatenate([iid_deltas, label_deltas]))
		ax.set_ylim(bottom=0, top=max_delta * 1.15 if not np.isnan(max_delta) else 50)
		
		plt.tight_layout()
		plt.savefig(self.output_root / filename, dpi=160, bbox_inches="tight")
		plt.close()

	def run(self) -> None:
		self.load()
		self.plot_all_18x2()
		self.plot_per_strategy()
		self.plot_grouped()
		self.plot_bar_chart()
		self.plot_bar_chart_sorted_by_iid()
		self.plot_bar_chart_sorted_by_label()
		self.make_table_grouped_by_category()
		self.make_tables()
		self.make_best_vs_final_tables()
		self.plot_best_vs_final_bar_chart_sorted_by_delta()
		self.plot_best_vs_final_bar_chart_by_distribution("iid")
		self.plot_best_vs_final_bar_chart_by_distribution("label")
		print(f"Saved outputs to: {self.output_root}")


if __name__ == "__main__":
	station_dir = Path(__file__).resolve().parent
	if station_dir.name != SECOND_STATION_NAME:
		print(f"Warning: current folder is '{station_dir.name}', expected '{SECOND_STATION_NAME}'")
	comparator = IIDLabelComparator(station_dir)
	comparator.run()
