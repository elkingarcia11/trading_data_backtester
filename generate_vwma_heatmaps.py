#!/usr/bin/env python3
"""
Generate VWMA Fast vs VWMA Slow heatmaps for options trading results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to non-interactive backend to avoid display issues
plt.ioff()

def create_vwma_heatmaps(csv_file_path, output_dir="data/analysis/vwma_heatmaps"):
    """
    Create heatmaps for VWMA fast vs VWMA slow combinations across specified metrics
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Filter for VWMA fast and slow data
    if 'vwma_fast' not in df.columns or 'vwma_slow' not in df.columns:
        print("ERROR: No vwma_fast or vwma_slow columns found in the data")
        return
    
    print(f"Loaded {len(df)} combinations")
    print(f"VWMA Fast range: {df['vwma_fast'].min()}-{df['vwma_fast'].max()}")
    print(f"VWMA Slow range: {df['vwma_slow'].min()}-{df['vwma_slow'].max()}")
    
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
    
    # Get unique VWMA values for creating the heatmap grid
    vwma_fast_values = sorted(df['vwma_fast'].unique())
    vwma_slow_values = sorted(df['vwma_slow'].unique())
    
    print(f"VWMA Fast values: {vwma_fast_values}")
    print(f"VWMA Slow values: {vwma_slow_values}")
    print(f"Creating heatmaps for {len(metrics)} metrics...")
    
    # Create heatmaps for each metric
    for metric_col, metric_info in metrics.items():
        if metric_col not in df.columns:
            print(f"Warning: Column '{metric_col}' not found in data, skipping...")
            continue
            
        print(f"  Creating heatmap for {metric_info['title']}...")
        
        # Create pivot table for the heatmap
        # VWMA Fast on x-axis (columns), VWMA Slow on y-axis (index)
        heatmap_data = df.pivot_table(
            values=metric_col,
            index='vwma_slow',
            columns='vwma_fast',
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
        plt.title(f'VWMA Fast vs VWMA Slow Heatmap: {metric_info["title"]}\n{metric_info["description"]}', 
                 fontsize=18, fontweight='bold', pad=25)
        plt.xlabel('VWMA Fast Period', fontsize=14, fontweight='bold')
        plt.ylabel('VWMA Slow Period', fontsize=14, fontweight='bold')
        
        # Improve tick labels
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        filename = f"{output_dir}/vwma_heatmap_{metric_col.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Saved: {filename}")
        
        # Clear the plot (don't show to avoid interactive issues)
        plt.close()
    
    # Create a comprehensive summary analysis
    print("\nCreating detailed analysis...")
    
    # Find best combinations for each metric
    print("\n🎯 BEST VWMA FAST vs SLOW COMBINATIONS BY METRIC:")
    print("=" * 60)
    
    best_combinations = {}
    
    for metric_col, metric_info in metrics.items():
        if metric_col in df.columns:
            # Find the best combination for this metric
            best_idx = df[metric_col].idxmax()
            best_combo = df.loc[best_idx]
            best_combinations[metric_col] = best_combo
            
            print(f"\n📊 {metric_info['title']}:")
            print(f"   VWMA Fast: {best_combo['vwma_fast']}, VWMA Slow: {best_combo['vwma_slow']}")
            if metric_col == 'win_rate':
                print(f"   Value: {best_combo[metric_col]:.1%}")
            else:
                print(f"   Value: {best_combo[metric_col]:.5f}")
            print(f"   Total Trades: {best_combo['total_trades']}")
            print(f"   Win Rate: {best_combo['win_rate']:.1%}")
    
    # Find combinations that appear in top 10 for multiple metrics
    print(f"\n🏆 CONSISTENTLY HIGH PERFORMING VWMA COMBINATIONS:")
    print("=" * 60)
    
    top_performers = {}
    for metric_col, metric_info in metrics.items():
        if metric_col in df.columns:
            top_10 = df.nlargest(10, metric_col)[['vwma_fast', 'vwma_slow', metric_col]]
            for _, row in top_10.iterrows():
                combo_key = f"VWMA_Fast_{int(row['vwma_fast'])}_Slow_{int(row['vwma_slow'])}"
                if combo_key not in top_performers:
                    top_performers[combo_key] = {'count': 0, 'metrics': []}
                top_performers[combo_key]['count'] += 1
                top_performers[combo_key]['metrics'].append(metric_info['title'])
    
    # Show combinations that appear in multiple top 10 lists
    multi_metric_performers = {k: v for k, v in top_performers.items() if v['count'] > 1}
    if multi_metric_performers:
        for combo, info in sorted(multi_metric_performers.items(), 
                                 key=lambda x: x[1]['count'], reverse=True):
            vwma_fast_val = int(combo.split('_')[2])
            vwma_slow_val = int(combo.split('_')[4])
            combo_data = df[(df['vwma_fast'] == vwma_fast_val) & (df['vwma_slow'] == vwma_slow_val)].iloc[0]
            print(f"\n   VWMA Fast {vwma_fast_val} × Slow {vwma_slow_val} (Top 10 in {info['count']} metrics):")
            print(f"     Metrics: {', '.join(info['metrics'])}")
            print(f"     Avg Profit: {combo_data['average_trade_profit']:.5f}")
            print(f"     Win Rate: {combo_data['win_rate']:.1%}")
            print(f"     Total Profit: {combo_data['total_trade_profit']:.3f}")
    else:
        print("   No combinations found in multiple top 10 lists")
    
    # Create a summary matrix showing rank of each combination across metrics
    print(f"\n📈 CREATING SUMMARY STATISTICS...")
    summary_stats = df.groupby(['vwma_fast', 'vwma_slow']).agg({
        'average_trade_profit': 'mean',
        'win_rate': 'mean',
        'total_trade_profit': 'mean',
        'total_trades': 'mean'
    }).round(5)
    
    # Save summary to CSV
    summary_file = f"{output_dir}/vwma_combinations_summary.csv"
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
        index='vwma_slow',
        columns='vwma_fast',
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
    
    plt.title('VWMA Fast vs Slow Composite Performance Score\n(Average of normalized: Avg Profit + Win Rate + Total Profit)', 
             fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('VWMA Fast Period', fontsize=14, fontweight='bold')
    plt.ylabel('VWMA Slow Period', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    
    composite_filename = f"{output_dir}/vwma_composite_score_heatmap.png"
    plt.savefig(composite_filename, dpi=300, bbox_inches='tight')
    print(f"Composite score heatmap saved: {composite_filename}")
    plt.close()
    
    # Find the best overall combination from composite score
    best_composite_idx = df_normalized['composite_score'].idxmax()
    best_composite = df_normalized.loc[best_composite_idx]
    
    print(f"\n🏆 BEST OVERALL VWMA COMBINATION (Composite Score):")
    print(f"   VWMA Fast: {best_composite['vwma_fast']}, VWMA Slow: {best_composite['vwma_slow']}")
    print(f"   Composite Score: {best_composite['composite_score']:.3f}")
    print(f"   Avg Profit: {best_composite['average_trade_profit']:.5f}")
    print(f"   Win Rate: {best_composite['win_rate']:.1%}")
    print(f"   Total Profit: {best_composite['total_trade_profit']:.3f}")
    print(f"   Total Trades: {best_composite['total_trades']}")
    
    print(f"\n✅ All VWMA Fast vs Slow heatmaps saved to: {output_dir}/")
    print(f"📊 {len(metrics) + 1} heatmaps created successfully!")

def main():
    """Main function to run the VWMA fast vs slow heatmap generation"""
    
    # File path to the options results
    csv_file = "data/results/options/SPY_unknown_unknown.csv"
    
    if not Path(csv_file).exists():
        print(f"ERROR: File {csv_file} not found!")
        print("Please make sure you have run the options backtesting first.")
        return
    
    # Generate heatmaps
    create_vwma_heatmaps(csv_file)

if __name__ == "__main__":
    main()