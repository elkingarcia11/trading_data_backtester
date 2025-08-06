#!/usr/bin/env python3
"""
Generate EMA vs VWMA heatmaps for options trading results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_ema_vwma_heatmaps(csv_file_path, output_dir="data/analysis/ema_vwma_heatmaps"):
    """
    Create heatmaps for EMA vs VWMA combinations across specified metrics
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Filter for EMA and VWMA data
    if 'ema' not in df.columns or 'vwma' not in df.columns:
        print("ERROR: No ema or vwma columns found in the data")
        return
    
    print(f"Loaded {len(df)} combinations")
    print(f"EMA range: {df['ema'].min()}-{df['ema'].max()}")
    print(f"VWMA range: {df['vwma'].min()}-{df['vwma'].max()}")
    
    # Define the three requested metrics
    metrics = {
        'average_trade_profit': {
            'title': 'Average Trade Profit',
            'cmap': 'RdYlGn',
            'fmt': '.5f',
            'description': 'Profit per trade on average'
        },
        'win_rate': {
            'title': 'Win Rate (%)',
            'cmap': 'RdYlGn',
            'fmt': '.1%',
            'description': 'Percentage of winning trades'
        },
        'total_trade_profit': {
            'title': 'Total Trade Profit',
            'cmap': 'RdYlGn',
            'fmt': '.3f',
            'description': 'Cumulative profit from all trades'
        }
    }
    
    # Get unique EMA and VWMA values for creating the heatmap grid
    ema_values = sorted(df['ema'].unique())
    vwma_values = sorted(df['vwma'].unique())
    
    print(f"EMA values: {ema_values}")
    print(f"VWMA values: {vwma_values}")
    print(f"Creating heatmaps for {len(metrics)} metrics...")
    
    # Create heatmaps for each metric
    for metric_col, metric_info in metrics.items():
        if metric_col not in df.columns:
            print(f"Warning: Column '{metric_col}' not found in data, skipping...")
            continue
            
        print(f"  Creating heatmap for {metric_info['title']}...")
        
        # Create pivot table for the heatmap
        # EMA on x-axis (columns), VWMA on y-axis (index)
        heatmap_data = df.pivot_table(
            values=metric_col,
            index='vwma',
            columns='ema',
            aggfunc='mean'  # In case there are duplicates, take the mean
        )
        
        # Create the heatmap
        plt.figure(figsize=(16, 12))
        
        # Create heatmap with custom formatting
        if metric_col == 'win_rate':
            # Convert win rate to percentage for better display
            sns.heatmap(
                heatmap_data * 100,
                annot=True,
                fmt='.1f',
                cmap=metric_info['cmap'],
                cbar_kws={'label': 'Win Rate (%)'},
                linewidths=0.5,
                square=False
            )
        else:
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=metric_info['fmt'],
                cmap=metric_info['cmap'],
                cbar_kws={'label': metric_info['title']},
                linewidths=0.5,
                square=False
            )
        
        # Customize the plot
        plt.title(f'EMA vs VWMA Heatmap: {metric_info["title"]}\n{metric_info["description"]}', 
                 fontsize=18, fontweight='bold', pad=25)
        plt.xlabel('EMA Period', fontsize=14, fontweight='bold')
        plt.ylabel('VWMA Period', fontsize=14, fontweight='bold')
        
        # Improve tick labels
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        filename = f"{output_dir}/ema_vwma_heatmap_{metric_col.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        
        # Show the plot
        plt.show()
        
        # Clear the plot
        plt.clf()
    
    # Create a comprehensive summary analysis
    print("\nCreating detailed analysis...")
    
    # Find best combinations for each metric
    print("\n🎯 BEST EMA vs VWMA COMBINATIONS BY METRIC:")
    print("=" * 60)
    
    best_combinations = {}
    
    for metric_col, metric_info in metrics.items():
        if metric_col in df.columns:
            # Find the best combination for this metric
            best_idx = df[metric_col].idxmax()
            best_combo = df.loc[best_idx]
            best_combinations[metric_col] = best_combo
            
            print(f"\n📊 {metric_info['title']}:")
            print(f"   EMA: {best_combo['ema']}, VWMA: {best_combo['vwma']}")
            if metric_col == 'win_rate':
                print(f"   Value: {best_combo[metric_col]:.1%}")
            else:
                print(f"   Value: {best_combo[metric_col]:.5f}")
            print(f"   Total Trades: {best_combo['total_trades']}")
            print(f"   Win Rate: {best_combo['win_rate']:.1%}")
    
    # Find combinations that appear in top 10 for multiple metrics
    print(f"\n🏆 CONSISTENTLY HIGH PERFORMING COMBINATIONS:")
    print("=" * 60)
    
    top_performers = {}
    for metric_col, metric_info in metrics.items():
        if metric_col in df.columns:
            top_10 = df.nlargest(10, metric_col)[['ema', 'vwma', metric_col]]
            for _, row in top_10.iterrows():
                combo_key = f"EMA_{int(row['ema'])}_VWMA_{int(row['vwma'])}"
                if combo_key not in top_performers:
                    top_performers[combo_key] = {'count': 0, 'metrics': []}
                top_performers[combo_key]['count'] += 1
                top_performers[combo_key]['metrics'].append(metric_info['title'])
    
    # Show combinations that appear in multiple top 10 lists
    multi_metric_performers = {k: v for k, v in top_performers.items() if v['count'] > 1}
    if multi_metric_performers:
        for combo, info in sorted(multi_metric_performers.items(), 
                                 key=lambda x: x[1]['count'], reverse=True):
            ema_val = int(combo.split('_')[1])
            vwma_val = int(combo.split('_')[3])
            combo_data = df[(df['ema'] == ema_val) & (df['vwma'] == vwma_val)].iloc[0]
            print(f"\n   EMA {ema_val} × VWMA {vwma_val} (Top 10 in {info['count']} metrics):")
            print(f"     Metrics: {', '.join(info['metrics'])}")
            print(f"     Avg Profit: {combo_data['average_trade_profit']:.5f}")
            print(f"     Win Rate: {combo_data['win_rate']:.1%}")
            print(f"     Total Profit: {combo_data['total_trade_profit']:.3f}")
    else:
        print("   No combinations found in multiple top 10 lists")
    
    # Create a summary matrix showing rank of each combination across metrics
    print(f"\n📈 CREATING SUMMARY STATISTICS...")
    summary_stats = df.groupby(['ema', 'vwma']).agg({
        'average_trade_profit': 'mean',
        'win_rate': 'mean',
        'total_trade_profit': 'mean',
        'total_trades': 'mean'
    }).round(5)
    
    # Save summary to CSV
    summary_file = f"{output_dir}/ema_vwma_combinations_summary.csv"
    summary_stats.to_csv(summary_file)
    print(f"Summary statistics saved to: {summary_file}")
    
    # Create a composite score heatmap
    print(f"\n🎯 Creating composite performance score heatmap...")
    
    # Normalize each metric to 0-1 scale for fair comparison
    df_normalized = df.copy()
    for metric_col in ['average_trade_profit', 'win_rate', 'total_trade_profit']:
        if metric_col in df.columns:
            min_val = df[metric_col].min()
            max_val = df[metric_col].max()
            df_normalized[f'{metric_col}_norm'] = (df[metric_col] - min_val) / (max_val - min_val)
    
    # Create composite score (equal weight for now)
    df_normalized['composite_score'] = (
        df_normalized['average_trade_profit_norm'] + 
        df_normalized['win_rate_norm'] + 
        df_normalized['total_trade_profit_norm']
    ) / 3
    
    # Create composite score heatmap
    composite_heatmap = df_normalized.pivot_table(
        values='composite_score',
        index='vwma',
        columns='ema',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        composite_heatmap,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Composite Performance Score (0-1)'},
        linewidths=0.5,
        square=False
    )
    
    plt.title('EMA vs VWMA Composite Performance Score\n(Average of normalized: Avg Profit + Win Rate + Total Profit)', 
             fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('EMA Period', fontsize=14, fontweight='bold')
    plt.ylabel('VWMA Period', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    
    composite_filename = f"{output_dir}/ema_vwma_composite_score_heatmap.png"
    plt.savefig(composite_filename, dpi=300, bbox_inches='tight')
    print(f"Composite score heatmap saved: {composite_filename}")
    plt.show()
    plt.clf()
    
    print(f"\n✅ All EMA vs VWMA heatmaps saved to: {output_dir}/")
    print(f"📊 {len(metrics) + 1} heatmaps created successfully!")

def main():
    """Main function to run the EMA vs VWMA heatmap generation"""
    
    # File path to the options results
    csv_file = "data/results/options/SPY_unknown_unknown.csv"
    
    if not Path(csv_file).exists():
        print(f"ERROR: File {csv_file} not found!")
        print("Please make sure you have run the options backtesting first.")
        return
    
    # Generate heatmaps
    create_ema_vwma_heatmaps(csv_file)

if __name__ == "__main__":
    main()