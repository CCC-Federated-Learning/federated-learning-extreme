import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

FIRST_STATION_NAME = "res-drag-set-to-there----first-station"

# Configure matplotlib for better rendering
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Define strategy groups and metadata
STRATEGY_GROUPS = {
    'Basic & Classic Strategies': ['FedAvg', 'FedAvgM', 'FedProx'],
    'Adaptive Optimization Strategies': ['FedAdagrad', 'FedAdam', 'FedYogi'],
    'Robustness & Byzantine Fault Tolerance': ['Bulyan', 'Krum', 'MultiKrum'],
    'Statistical Robustness Strategies': ['FedMedian', 'FedTrimmedAvg'],
    'Differential Privacy Strategies': [
        'DifferentialPrivacyClientSideAdaptiveClipping',
        'DifferentialPrivacyClientSideFixedClipping',
        'DifferentialPrivacyServerSideAdaptiveClipping',
        'DifferentialPrivacyServerSideFixedClipping'
    ],
    'XGBoost Specific Strategies': ['FedXgbBagging', 'FedXgbCyclic'],
    'Fairness & Weighted Optimization': ['QFedAvg']
}

# Color palette for different strategies
COLORS = {
    'FedAvg': '#1f77b4', 'FedAvgM': '#ff7f0e', 'FedProx': '#2ca02c',
    'FedAdagrad': '#d62728', 'FedAdam': '#9467bd', 'FedYogi': '#8c564b',
    'Bulyan': '#e377c2', 'Krum': '#7f7f7f', 'MultiKrum': '#bcbd22',
    'FedMedian': '#17becf', 'FedTrimmedAvg': '#1f77b4',
    'DifferentialPrivacyClientSideAdaptiveClipping': '#ff9999',
    'DifferentialPrivacyClientSideFixedClipping': '#ffcc99',
    'DifferentialPrivacyServerSideAdaptiveClipping': '#99ccff',
    'DifferentialPrivacyServerSideFixedClipping': '#cc99ff',
    'FedXgbBagging': '#99ff99', 'FedXgbCyclic': '#ffff99',
    'QFedAvg': '#00ffff'
}

DP_SHORT_NAMES = {
    'DifferentialPrivacyClientSideAdaptiveClipping': 'DP-Client-Adaptive',
    'DifferentialPrivacyClientSideFixedClipping': 'DP-Client-Fixed',
    'DifferentialPrivacyServerSideAdaptiveClipping': 'DP-Server-Adaptive',
    'DifferentialPrivacyServerSideFixedClipping': 'DP-Server-Fixed',
}


class StrategyAnalyzer:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.strategy_data = {}
        self._load_all_data()
        self.chart_context = self._build_chart_context()

    @staticmethod
    def _metadata_value(content: str, key: str) -> str:
        match = re.search(rf'{re.escape(key)}\s*=\s*(.+)', content)
        return match.group(1).strip() if match else ''

    @staticmethod
    def _normalize_distribution(distribution: str) -> str:
        if not distribution:
            return ''
        normalized = distribution.strip().lower()
        mapping = {
            'iid': 'IID',
            'label': 'LABEL',
            'dirichlet': 'DIRICHLET'
        }
        return mapping.get(normalized, distribution.strip().upper())

    @staticmethod
    def _format_maybe_mixed(values: List[str], label: str) -> str:
        if not values:
            return f"{label}: Unknown"
        unique_values = sorted(set(v for v in values if v))
        if not unique_values:
            return f"{label}: Unknown"
        if len(unique_values) == 1:
            return f"{label}: {unique_values[0]}"
        return f"{label}: Mixed ({', '.join(unique_values)})"

    def _build_chart_context(self) -> str:
        distributions = [data.get('distribution', '') for data in self.strategy_data.values()]
        datasets = [data.get('dataset_name', '') for data in self.strategy_data.values()]
        rounds = [str(data.get('num_rounds', '')) for data in self.strategy_data.values() if data.get('num_rounds')]
        epochs = [str(data.get('local_epochs', '')) for data in self.strategy_data.values() if data.get('local_epochs')]

        parts = [
            self._format_maybe_mixed(datasets, 'Dataset'),
            self._format_maybe_mixed(distributions, 'Distribution'),
            self._format_maybe_mixed(rounds, 'Rounds'),
            self._format_maybe_mixed(epochs, 'Local Epochs')
        ]
        return ' | '.join(parts)

    def _apply_context(self, fig):
        fig.text(
            0.01,
            0.01,
            self.chart_context,
            ha='left',
            va='bottom',
            fontsize=9,
            color='#444444',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#f5f5f5', edgecolor='#cccccc', alpha=0.8)
        )

    def _load_all_data(self):
        """Load metrics and metadata for all strategies."""
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            metadata_path = subdir / 'metadata.txt'
            metrics_path = subdir / 'run_metrics.csv'
            
            if not (metadata_path.exists() and metrics_path.exists()):
                continue
            
            # Parse metadata
            strategy = None
            elapsed_seconds = 0
            distribution = ''
            dataset_name = ''
            num_rounds = ''
            local_epochs = ''
            with open(metadata_path, 'r') as f:
                content = f.read()
                match = re.search(r'strategy = (\w+)', content)
                if match:
                    strategy = match.group(1)
                elapsed_match = re.search(r'elapsed_seconds = ([\d.]+)', content)
                if elapsed_match:
                    elapsed_seconds = float(elapsed_match.group(1))
                distribution = self._normalize_distribution(self._metadata_value(content, 'data_distribution'))
                dataset_name = self._metadata_value(content, 'dataset_name')
                num_rounds = self._metadata_value(content, 'num_rounds')
                local_epochs = self._metadata_value(content, 'local_epochs')
            
            if not strategy:
                continue
            
            # Load metrics
            df = pd.read_csv(metrics_path)
            
            self.strategy_data[strategy] = {
                'metrics': df,
                'time': elapsed_seconds,
                'subdir': subdir,
                'distribution': distribution,
                'dataset_name': dataset_name,
                'num_rounds': num_rounds,
                'local_epochs': local_epochs
            }

    def plot_all_strategies_acc_round(self):
        """Plot accuracy vs round for all 18 strategies."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for strategy, data in sorted(self.strategy_data.items()):
            df = data['metrics']
            ax.plot(df['round'], df['accuracy'], marker='o', 
                   color=COLORS.get(strategy, '#000000'),
                     label=strategy, linewidth=1.3, markersize=3, alpha=0.8)
        
        ax.set_xlabel('Training Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(
                f'Comprehensive Strategy Comparison: Accuracy over Training Rounds\n'
                f'(18 Federated Learning Strategies)\n{self.chart_context}',
                    fontsize=13, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        self._apply_context(fig)
        
        plt.tight_layout()
        output_path = self.output_dir / '01_acc_round_all_strategies.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_accuracy_time_table(self):
        """Create a comparison table of accuracy and training time."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        table_data = []
        strategies_list = sorted(self.strategy_data.keys())
        
        for strategy in strategies_list:
            data = self.strategy_data[strategy]
            max_acc = data['metrics']['accuracy'].max() * 100
            time = data['time']
            table_data.append([strategy, f"{max_acc:.2f}%", f"{time:.2f}s"])
        
        # Sort by accuracy descending to find top 3
        table_data_sorted = sorted(table_data, key=lambda x: float(x[1].rstrip('%')), reverse=True)
        
        # Find top 3 in each category
        top_3_acc = set(row[0] for row in table_data_sorted[:3])
        top_3_time = set(strategies_list[i] for i in np.argsort([self.strategy_data[s]['time'] for s in strategies_list])[:3])
        
        # Create colors for cells
        cell_colors = []
        for row in table_data:
            strategy = row[0]
            colors = []
            colors.append('#ffffff')  # strategy name
            
            # Accuracy cell - red if top 3
            acc_color = '#ffcccc' if strategy in top_3_acc else '#ffffff'
            colors.append(acc_color)
            
            # Time cell - blue if top 3 (fastest)
            time_color = '#ccccff' if strategy in top_3_time else '#ffffff'
            colors.append(time_color)
            
            cell_colors.append(colors)
        
        columns = ['Strategy', 'Best Accuracy', 'Training Time']
        table = ax.table(cellText=table_data, colLabels=columns, 
                        cellLoc='center', loc='center',
                        cellColours=cell_colors,
                        colColours=['#e6e6e6']*3)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#999999')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add legend
        fig.text(0.5, 0.05, 
                'Red shading: Top 3 Best Accuracy  |  Blue shading: Top 3 Fastest (Total Training Time)',
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle(f'Strategy Performance Table: Best Accuracy vs Training Time\n(All 18 Strategies)\n{self.chart_context}', 
                    fontsize=13, fontweight='bold', y=0.98)
        self._apply_context(fig)
        
        output_path = self.output_dir / '02a_accuracy_time_comparison_table.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_grouped_strategies(self):
        """Plot accuracy vs round for each strategy group."""
        num_groups = len(STRATEGY_GROUPS)
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        axes = axes.flatten()
        
        group_idx = 0
        for group_name, strategies in STRATEGY_GROUPS.items():
            ax = axes[group_idx]
            
            # Filter available strategies in this group
            available_strategies = [s for s in strategies if s in self.strategy_data]
            
            if not available_strategies:
                ax.text(0.5, 0.5, f"{group_name}\n(No data)", 
                       ha='center', va='center', fontsize=11)
                ax.axis('off')
                group_idx += 1
                continue
            
            # Plot each strategy in this group
            for strategy in available_strategies:
                data = self.strategy_data[strategy]
                df = data['metrics']
                
                # Use shorthand names for DP strategies
                display_name = DP_SHORT_NAMES.get(strategy, strategy)
                
                ax.plot(df['round'], df['accuracy'], marker='o',
                       color=COLORS.get(strategy, '#000000'),
                      label=display_name, linewidth=1.6, markersize=3, alpha=0.8)
            
            ax.set_xlabel('Training Round', fontsize=10, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
            ax.set_title(f'{group_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='best')
            
            group_idx += 1
        
        # Hide extra subplots
        for idx in range(group_idx, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'Grouped Strategy Analysis: Accuracy over Training Rounds\n{self.chart_context}',
                    fontsize=14, fontweight='bold', y=0.995)
        self._apply_context(fig)
        
        plt.tight_layout()
        output_path = self.output_dir / '03a_grouped_strategies_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_grouped_subplots(self):
        """Plot smaller grouped comparison subplots."""
        num_groups = len(STRATEGY_GROUPS)
        ncols = 2
        nrows = (num_groups + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5*nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        group_idx = 0
        for group_name, strategies in STRATEGY_GROUPS.items():
            ax = axes[group_idx]
            
            available_strategies = [s for s in strategies if s in self.strategy_data]
            
            if not available_strategies:
                ax.text(0.5, 0.5, f"{group_name}\n(No data)", 
                       ha='center', va='center', fontsize=11)
                ax.axis('off')
                group_idx += 1
                continue
            
            for strategy in available_strategies:
                data = self.strategy_data[strategy]
                df = data['metrics']
                display_name = DP_SHORT_NAMES.get(strategy, strategy)
                
                ax.plot(df['round'], df['accuracy'], marker='s',
                       color=COLORS.get(strategy, '#000000'),
                      label=display_name, linewidth=1.6, markersize=3, alpha=0.85)
            
            ax.set_xlabel('Round', fontsize=9, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=9, fontweight='bold')
            ax.set_title(f'{group_name}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.legend(fontsize=8, loc='best')
            
            group_idx += 1
        
        for idx in range(group_idx, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'Grouped Comparison of Strategies by Category\n{self.chart_context}',
                    fontsize=12, fontweight='bold', y=0.995)
        self._apply_context(fig)
        
        plt.tight_layout()
        output_path = self.output_dir / '03b_grouped_detailed_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_final_accuracy_boxplot(self):
        """Plot final accuracy comparison as bar chart grouped by category."""
        fig, ax = plt.subplots(figsize=(15, 7))
        
        positions = []
        final_accs = []
        colors_list = []
        labels = []
        pos = 0
        
        group_spacing = 1.5
        
        for group_name, strategies in STRATEGY_GROUPS.items():
            available_strategies = [s for s in strategies if s in self.strategy_data]
            
            if not available_strategies:
                continue
            
            for strategy in available_strategies:
                data = self.strategy_data[strategy]
                final_acc = data['metrics']['accuracy'].iloc[-1] * 100
                
                positions.append(pos)
                final_accs.append(final_acc)
                colors_list.append(COLORS.get(strategy, '#000000'))
                
                display_label = DP_SHORT_NAMES.get(strategy, strategy)
                labels.append(display_label)
                pos += 1
            
            pos += group_spacing - 1
        
        bars = ax.bar(positions, final_accs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Final Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Final Model Accuracy Comparison: All 18 Strategies\n{self.chart_context}',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])
        self._apply_context(fig)
        
        plt.tight_layout()
        output_path = self.output_dir / '04a_final_accuracy_by_group.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_accuracy_time_table_sorted(self):
        """Create a comparison table sorted by accuracy (highest to lowest)."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        table_data = []
        for strategy in sorted(self.strategy_data.keys()):
            data = self.strategy_data[strategy]
            max_acc = data['metrics']['accuracy'].max() * 100
            time = data['time']
            table_data.append([strategy, f"{max_acc:.2f}%", f"{time:.2f}s"])
        
        # Sort by accuracy descending
        table_data_sorted = sorted(table_data, key=lambda x: float(x[1].rstrip('%')), reverse=True)
        
        # Find top 3 in each category
        top_3_acc = set(row[0] for row in table_data_sorted[:3])
        strategies_list = [row[0] for row in table_data_sorted]
        times_sorted = sorted([(row[0], float(row[2].rstrip('s'))) for row in table_data_sorted], key=lambda x: x[1])
        top_3_time = set(s for s, _ in times_sorted[:3])
        
        # Create colors for cells
        cell_colors = []
        for row in table_data_sorted:
            strategy = row[0]
            colors = []
            colors.append('#ffffff')  # strategy name
            
            # Accuracy cell - red if top 3
            acc_color = '#ffcccc' if strategy in top_3_acc else '#ffffff'
            colors.append(acc_color)
            
            # Time cell - blue if top 3 (fastest)
            time_color = '#ccccff' if strategy in top_3_time else '#ffffff'
            colors.append(time_color)
            
            cell_colors.append(colors)
        
        columns = ['Strategy', 'Best Accuracy', 'Training Time']
        table = ax.table(cellText=table_data_sorted, colLabels=columns, 
                        cellLoc='center', loc='center',
                        cellColours=cell_colors,
                        colColours=['#e6e6e6']*3)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#999999')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add legend
        fig.text(0.5, 0.05, 
                'Sorted by Accuracy (Highest to Lowest) | Red shading: Top 3 Best Accuracy  |  Blue shading: Top 3 Fastest',
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle(f'Strategy Performance Table (Sorted by Accuracy)\nBest Accuracy vs Training Time - All 18 Strategies\n{self.chart_context}', 
                    fontsize=13, fontweight='bold', y=0.98)
        self._apply_context(fig)
        
        output_path = self.output_dir / '02b_accuracy_time_comparison_table_sorted.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_accuracy_time_table_sorted_by_time(self):
        """Create a comparison table sorted by training time (fastest to slowest)."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        table_data = []
        for strategy in sorted(self.strategy_data.keys()):
            data = self.strategy_data[strategy]
            max_acc = data['metrics']['accuracy'].max() * 100
            time = data['time']
            table_data.append([strategy, f"{max_acc:.2f}%", f"{time:.2f}s"])
        
        # Sort by time ascending (fastest first)
        table_data_sorted = sorted(table_data, key=lambda x: float(x[2].rstrip('s')))
        
        # Find top 3 in each category
        acc_sorted = sorted([(row[0], float(row[1].rstrip('%'))) for row in table_data_sorted], key=lambda x: x[1], reverse=True)
        top_3_acc = set(s for s, _ in acc_sorted[:3])
        top_3_time = set(row[0] for row in table_data_sorted[:3])
        
        # Create colors for cells
        cell_colors = []
        for row in table_data_sorted:
            strategy = row[0]
            colors = []
            colors.append('#ffffff')  # strategy name
            
            # Accuracy cell - red if top 3
            acc_color = '#ffcccc' if strategy in top_3_acc else '#ffffff'
            colors.append(acc_color)
            
            # Time cell - blue if top 3 (fastest)
            time_color = '#ccccff' if strategy in top_3_time else '#ffffff'
            colors.append(time_color)
            
            cell_colors.append(colors)
        
        columns = ['Strategy', 'Best Accuracy', 'Training Time']
        table = ax.table(cellText=table_data_sorted, colLabels=columns, 
                        cellLoc='center', loc='center',
                        cellColours=cell_colors,
                        colColours=['#e6e6e6']*3)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#999999')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add legend
        fig.text(0.5, 0.05, 
                'Sorted by Training Time (Fastest to Slowest) | Red shading: Top 3 Best Accuracy  |  Blue shading: Top 3 Fastest Training Time',
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle(f'Strategy Performance Table (Sorted by Training Time)\nBest Accuracy vs Training Time - All 18 Strategies\n{self.chart_context}', 
                    fontsize=13, fontweight='bold', y=0.98)
        self._apply_context(fig)
        
        output_path = self.output_dir / '02c_accuracy_time_comparison_table_sorted_by_time.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_final_accuracy_sorted(self):
        """Plot final accuracy comparison sorted by accuracy (highest to lowest)."""
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Collect all strategies with their final accuracy
        strategy_acc_list = []
        for strategy in self.strategy_data.keys():
            data = self.strategy_data[strategy]
            final_acc = data['metrics']['accuracy'].iloc[-1] * 100
            strategy_acc_list.append((strategy, final_acc))
        
        # Sort by accuracy descending
        strategy_acc_list.sort(key=lambda x: x[1], reverse=True)
        
        positions = list(range(len(strategy_acc_list)))
        final_accs = [acc for _, acc in strategy_acc_list]
        labels = []
        colors_list = []
        
        for strategy, _ in strategy_acc_list:
            display_label = DP_SHORT_NAMES.get(strategy, strategy)
            labels.append(display_label)
            colors_list.append(COLORS.get(strategy, '#000000'))
        
        bars = ax.bar(positions, final_accs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Final Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Final Model Accuracy Comparison (Sorted by Accuracy)\nAll 18 Strategies - Highest to Lowest Performance\n{self.chart_context}',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])
        self._apply_context(fig)
        
        plt.tight_layout()
        output_path = self.output_dir / '04b_final_accuracy_sorted.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_final_round_accuracy_time_table(self):
        """Create a comparison table of final round accuracy and training time."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data using final round accuracy
        table_data = []
        strategies_list = sorted(self.strategy_data.keys())
        
        for strategy in strategies_list:
            data = self.strategy_data[strategy]
            final_round_acc = data['metrics']['accuracy'].iloc[-1] * 100
            time = data['time']
            table_data.append([strategy, f"{final_round_acc:.2f}%", f"{time:.2f}s"])
        
        # Sort by final round accuracy descending to find top 3
        table_data_sorted = sorted(table_data, key=lambda x: float(x[1].rstrip('%')), reverse=True)
        
        # Find top 3 in each category
        top_3_acc = set(row[0] for row in table_data_sorted[:3])
        top_3_time = set(strategies_list[i] for i in np.argsort([self.strategy_data[s]['time'] for s in strategies_list])[:3])
        
        # Create colors for cells
        cell_colors = []
        for row in table_data:
            strategy = row[0]
            colors = []
            colors.append('#ffffff')  # strategy name
            
            # Accuracy cell - red if top 3
            acc_color = '#ffcccc' if strategy in top_3_acc else '#ffffff'
            colors.append(acc_color)
            
            # Time cell - blue if top 3 (fastest)
            time_color = '#ccccff' if strategy in top_3_time else '#ffffff'
            colors.append(time_color)
            
            cell_colors.append(colors)
        
        columns = ['Strategy', 'Final Round Accuracy', 'Training Time']
        table = ax.table(cellText=table_data, colLabels=columns, 
                        cellLoc='center', loc='center',
                        cellColours=cell_colors,
                        colColours=['#e6e6e6']*3)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#999999')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add legend
        fig.text(0.5, 0.05, 
                'Red shading: Top 3 Final Round Accuracy  |  Blue shading: Top 3 Fastest (Total Training Time)',
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle(f'Strategy Performance Table (Final Round Accuracy): Final Round Accuracy vs Training Time\n(All 18 Strategies)\n{self.chart_context}', 
                    fontsize=13, fontweight='bold', y=0.98)
        self._apply_context(fig)
        
        output_path = self.output_dir / '05a_final_round_accuracy_time_comparison_table.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_final_round_accuracy_time_table_sorted(self):
        """Create a comparison table sorted by final round accuracy (highest to lowest)."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data using final round accuracy
        table_data = []
        for strategy in sorted(self.strategy_data.keys()):
            data = self.strategy_data[strategy]
            final_round_acc = data['metrics']['accuracy'].iloc[-1] * 100
            time = data['time']
            table_data.append([strategy, f"{final_round_acc:.2f}%", f"{time:.2f}s"])
        
        # Sort by final round accuracy descending
        table_data_sorted = sorted(table_data, key=lambda x: float(x[1].rstrip('%')), reverse=True)
        
        # Find top 3 in each category
        top_3_acc = set(row[0] for row in table_data_sorted[:3])
        strategies_list = [row[0] for row in table_data_sorted]
        times_sorted = sorted([(row[0], float(row[2].rstrip('s'))) for row in table_data_sorted], key=lambda x: x[1])
        top_3_time = set(s for s, _ in times_sorted[:3])
        
        # Create colors for cells
        cell_colors = []
        for row in table_data_sorted:
            strategy = row[0]
            colors = []
            colors.append('#ffffff')  # strategy name
            
            # Accuracy cell - red if top 3
            acc_color = '#ffcccc' if strategy in top_3_acc else '#ffffff'
            colors.append(acc_color)
            
            # Time cell - blue if top 3 (fastest)
            time_color = '#ccccff' if strategy in top_3_time else '#ffffff'
            colors.append(time_color)
            
            cell_colors.append(colors)
        
        columns = ['Strategy', 'Final Round Accuracy', 'Training Time']
        table = ax.table(cellText=table_data_sorted, colLabels=columns, 
                        cellLoc='center', loc='center',
                        cellColours=cell_colors,
                        colColours=['#e6e6e6']*3)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#999999')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add legend
        fig.text(0.5, 0.05, 
                'Sorted by Final Round Accuracy (Highest to Lowest) | Red shading: Top 3 Final Round Accuracy  |  Blue shading: Top 3 Fastest',
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle(f'Strategy Performance Table (Sorted by Final Round Accuracy)\nFinal Round Accuracy vs Training Time - All 18 Strategies\n{self.chart_context}', 
                    fontsize=13, fontweight='bold', y=0.98)
        self._apply_context(fig)
        
        output_path = self.output_dir / '05b_final_round_accuracy_time_comparison_table_sorted.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_final_round_accuracy_time_table_sorted_by_time(self):
        """Create a comparison table sorted by training time (fastest to slowest) - using final round accuracy."""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data using final round accuracy
        table_data = []
        for strategy in sorted(self.strategy_data.keys()):
            data = self.strategy_data[strategy]
            final_round_acc = data['metrics']['accuracy'].iloc[-1] * 100
            time = data['time']
            table_data.append([strategy, f"{final_round_acc:.2f}%", f"{time:.2f}s"])
        
        # Sort by time ascending (fastest first)
        table_data_sorted = sorted(table_data, key=lambda x: float(x[2].rstrip('s')))
        
        # Find top 3 in each category
        acc_sorted = sorted([(row[0], float(row[1].rstrip('%'))) for row in table_data_sorted], key=lambda x: x[1], reverse=True)
        top_3_acc = set(s for s, _ in acc_sorted[:3])
        top_3_time = set(row[0] for row in table_data_sorted[:3])
        
        # Create colors for cells
        cell_colors = []
        for row in table_data_sorted:
            strategy = row[0]
            colors = []
            colors.append('#ffffff')  # strategy name
            
            # Accuracy cell - red if top 3
            acc_color = '#ffcccc' if strategy in top_3_acc else '#ffffff'
            colors.append(acc_color)
            
            # Time cell - blue if top 3 (fastest)
            time_color = '#ccccff' if strategy in top_3_time else '#ffffff'
            colors.append(time_color)
            
            cell_colors.append(colors)
        
        columns = ['Strategy', 'Final Round Accuracy', 'Training Time']
        table = ax.table(cellText=table_data_sorted, colLabels=columns, 
                        cellLoc='center', loc='center',
                        cellColours=cell_colors,
                        colColours=['#e6e6e6']*3)
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(3):
            table[(0, i)].set_facecolor('#999999')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add legend
        fig.text(0.5, 0.05, 
                'Sorted by Training Time (Fastest to Slowest) | Red shading: Top 3 Final Round Accuracy  |  Blue shading: Top 3 Fastest Training Time',
                ha='center', fontsize=10, style='italic')
        
        fig.suptitle(f'Strategy Performance Table (Sorted by Training Time)\nFinal Round Accuracy vs Training Time - All 18 Strategies\n{self.chart_context}', 
                    fontsize=13, fontweight='bold', y=0.98)
        self._apply_context(fig)
        
        output_path = self.output_dir / '05c_final_round_accuracy_time_comparison_table_sorted_by_time.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_final_round_accuracy_boxplot(self):
        """Plot final round accuracy comparison as bar chart grouped by category."""
        fig, ax = plt.subplots(figsize=(15, 7))
        
        positions = []
        final_accs = []
        colors_list = []
        labels = []
        pos = 0
        
        group_spacing = 1.5
        
        for group_name, strategies in STRATEGY_GROUPS.items():
            available_strategies = [s for s in strategies if s in self.strategy_data]
            
            if not available_strategies:
                continue
            
            for strategy in available_strategies:
                data = self.strategy_data[strategy]
                final_round_acc = data['metrics']['accuracy'].iloc[-1] * 100
                
                positions.append(pos)
                final_accs.append(final_round_acc)
                colors_list.append(COLORS.get(strategy, '#000000'))
                
                display_label = DP_SHORT_NAMES.get(strategy, strategy)
                labels.append(display_label)
                pos += 1
            
            pos += group_spacing - 1
        
        bars = ax.bar(positions, final_accs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Final Round Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Final Round Model Accuracy Comparison: All 18 Strategies\n{self.chart_context}',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])
        self._apply_context(fig)
        
        plt.tight_layout()
        output_path = self.output_dir / '06a_final_round_accuracy_by_group.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def plot_final_round_accuracy_sorted(self):
        """Plot final round accuracy comparison sorted by accuracy (highest to lowest)."""
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Collect all strategies with their final round accuracy
        strategy_acc_list = []
        for strategy in self.strategy_data.keys():
            data = self.strategy_data[strategy]
            final_round_acc = data['metrics']['accuracy'].iloc[-1] * 100
            strategy_acc_list.append((strategy, final_round_acc))
        
        # Sort by accuracy descending
        strategy_acc_list.sort(key=lambda x: x[1], reverse=True)
        
        positions = list(range(len(strategy_acc_list)))
        final_accs = [acc for _, acc in strategy_acc_list]
        labels = []
        colors_list = []
        
        for strategy, _ in strategy_acc_list:
            display_label = DP_SHORT_NAMES.get(strategy, strategy)
            labels.append(display_label)
            colors_list.append(COLORS.get(strategy, '#000000'))
        
        bars = ax.bar(positions, final_accs, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Final Round Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Final Round Model Accuracy Comparison (Sorted by Accuracy)\nAll 18 Strategies - Highest to Lowest Performance\n{self.chart_context}',
                    fontsize=12, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 105])
        self._apply_context(fig)
        
        plt.tight_layout()
        output_path = self.output_dir / '06b_final_round_accuracy_sorted.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path.name}")
        plt.close()

    def generate_all_charts(self):
        """Generate all charts in sequence."""
        print("\n" + "="*60)
        print("Generating Strategy Comparison Charts")
        print("="*60)
        
        print("\n01. Generating all-strategies accuracy curve...")
        self.plot_all_strategies_acc_round()
        
        print("02a. Generating accuracy-time comparison table (best accuracy)...")
        self.plot_accuracy_time_table()
        
        print("02b. Generating sorted accuracy-time comparison table (by best accuracy)...")
        self.plot_accuracy_time_table_sorted()
        
        print("02c. Generating sorted accuracy-time comparison table (by time)...")
        self.plot_accuracy_time_table_sorted_by_time()
        
        print("03a. Generating grouped strategy analysis...")
        self.plot_grouped_strategies()
        
        print("03b. Generating detailed grouped comparison...")
        self.plot_grouped_subplots()
        
        print("04a. Generating final accuracy comparison...")
        self.plot_final_accuracy_boxplot()
        
        print("04b. Generating final accuracy comparison (sorted)...")
        self.plot_final_accuracy_sorted()
        
        print("05a. Generating final round accuracy-time comparison table...")
        self.plot_final_round_accuracy_time_table()
        
        print("05b. Generating sorted final round accuracy-time comparison table (by accuracy)...")
        self.plot_final_round_accuracy_time_table_sorted()
        
        print("05c. Generating sorted final round accuracy-time comparison table (by time)...")
        self.plot_final_round_accuracy_time_table_sorted_by_time()
        
        print("06a. Generating final round accuracy comparison...")
        self.plot_final_round_accuracy_boxplot()
        
        print("06b. Generating final round accuracy comparison (sorted)...")
        self.plot_final_round_accuracy_sorted()
        
        print("\n" + "="*60)
        print(f"✓ All charts saved to: {self.output_dir}")
        print("="*60 + "\n")


if __name__ == '__main__':
    # Set paths
    current_dir = Path(__file__).resolve().parent
    if current_dir.name != FIRST_STATION_NAME:
        print(f"Warning: current folder name is '{current_dir.name}', expected '{FIRST_STATION_NAME}'")
    data_dir = current_dir / 'PUT-DATA-THERE'
    output_dir = current_dir / 'generate_charts'
    
    # Create analyzer and generate charts
    analyzer = StrategyAnalyzer(str(data_dir), str(output_dir))
    analyzer.generate_all_charts()
