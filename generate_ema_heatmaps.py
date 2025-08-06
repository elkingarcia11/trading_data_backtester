#!/usr/bin/env python3
"""
Generate comprehensive heatmaps for EMA fast/slow combinations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_ema_heatmaps(csv_file_path, output_dir="ema_heatmaps"):
    """
    Create heatmaps for EMA fast/slow combinations across different metrics
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Filter for EMA data only (should have ema_fast and ema_slow columns)
    if 'ema_fast' not in df.columns or 'ema_slow' not in df.columns:
        print("ERROR: No ema_fast or ema_slow columns found in the data")
        return
    
    print(f"Loaded {len(df)} combinations")
    print(f"EMA Fast range: {df['ema_fast'].min()}-{df['ema_fast'].max()}")
    print(f"EMA Slow range: {df['ema_slow'].min()}-{df['ema_slow'].max()}")
    
    # Define metrics to create heatmaps for
    metrics = {
        'average_trade_profit': {
            'title': 'Average Trade Profit',
            'cmap': 'RdYlGn',
            'fmt': '.4f'
        },
        'win_rate': {
            'title': 'Win Rate (%)',
            'cmap': 'RdYlGn',
            'fmt': '.1%'
        },
        'total_trades': {
            'title': 'Total Number of Trades',
            'cmap': 'viridis',
            'fmt': '.0f'
        },
        'total_trade_profit': {
            'title': 'Total Trade Profit',
            'cmap': 'RdYlGn',
            'fmt': '.3f'
        },
        'average_trade_duration (minutes)': {
            'title': 'Average Trade Duration (minutes)',
            'cmap': 'plasma',
            'fmt': '.1f'
        },
        'average_max_drawdown': {
            'title': 'Average Max Drawdown',
            'cmap': 'RdYlBu_r',
            'fmt': '.4f'
        },
        'average_max_unrealized_profit': {
            'title': 'Average Max Unrealized Profit',
            'cmap': 'RdYlGn',
            'fmt': '.4f'
        }
    }
    
    # Get unique EMA values for creating the heatmap grid
    ema_fast_values = sorted(df['ema_fast'].unique())
    ema_slow_values = sorted(df['ema_slow'].unique())
    
    print(f"Creating heatmaps for {len(metrics)} metrics...")
    
    # Create heatmaps for each metric
    for metric_col, metric_info in metrics.items():
        if metric_col not in df.columns:
            print(f"Warning: Column '{metric_col}' not found in data, skipping...")
            continue
            
        print(f"  Creating heatmap for {metric_info['title']}...")
        
        # Create pivot table for the heatmap
        heatmap_data = df.pivot_table(
            values=metric_col,
            index='ema_slow',
            columns='ema_fast',
            aggfunc='mean'  # In case there are duplicates, take the mean
        )
        
        # Create the heatmap
        plt.figure(figsize=(14, 10))
        
        # Create heatmap with custom formatting
        if metric_col == 'win_rate':
            # Convert win rate to percentage for better display
            sns.heatmap(
                heatmap_data * 100,
                annot=True,
                fmt='.1f',
                cmap=metric_info['cmap'],
                cbar_kws={'label': 'Win Rate (%)'},
                linewidths=0.5
            )
        else:
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=metric_info['fmt'],
                cmap=metric_info['cmap'],
                cbar_kws={'label': metric_info['title']},
                linewidths=0.5
            )
        
        # Customize the plot
        plt.title(f'EMA Combinations Heatmap: {metric_info["title"]}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('EMA Fast Period', fontsize=12, fontweight='bold')
        plt.ylabel('EMA Slow Period', fontsize=12, fontweight='bold')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        filename = f"{output_dir}/ema_heatmap_{metric_col.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        
        # Show the plot
        plt.show()
        
        # Clear the plot
        plt.clf()
    
    # Create a summary statistics table
    print("\nCreating summary statistics...")
    summary_stats = df.groupby(['ema_fast', 'ema_slow']).agg({
        'average_trade_profit': 'mean',
        'win_rate': 'mean',
        'total_trades': 'mean',
        'total_trade_profit': 'mean',
        'average_trade_duration (minutes)': 'mean',
        'average_max_drawdown': 'mean',
        'average_max_unrealized_profit': 'mean'
    }).round(4)
    
    # Save summary to CSV
    summary_file = f"{output_dir}/ema_combinations_summary.csv"
    summary_stats.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Find best combinations for each metric
    print("\n🎯 BEST COMBINATIONS BY METRIC:")
    print("=" * 50)
    
    for metric_col, metric_info in metrics.items():
        if metric_col in df.columns:
            if metric_col == 'average_max_drawdown':
                # For drawdown, we want the least negative (closest to 0)
                best_combo = df.loc[df[metric_col].idxmax()]
            else:
                # For other metrics, we want the maximum
                best_combo = df.loc[df[metric_col].idxmax()]
                
            print(f"\n{metric_info['title']}:")
            print(f"  EMA Fast: {best_combo['ema_fast']}, EMA Slow: {best_combo['ema_slow']}")
            if metric_col == 'win_rate':
                print(f"  Value: {best_combo[metric_col]:.1%}")
            else:
                print(f"  Value: {best_combo[metric_col]:.4f}")
    
    print(f"\n✅ All heatmaps saved to: {output_dir}/")
    print(f"📊 {len(metrics)} heatmaps created successfully!")

def main():
    """Main function to run the heatmap generation"""
    
    # File path to the options results
    csv_file = "data/results/options/SPY_unknown_unknown.csv"
    
    if not Path(csv_file).exists():
        print(f"ERROR: File {csv_file} not found!")
        print("Please make sure you have run the options backtesting first.")
        return
    
    # Generate heatmaps
    create_ema_heatmaps(csv_file)

if __name__ == "__main__":
    main()