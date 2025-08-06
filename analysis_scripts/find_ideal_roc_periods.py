#!/usr/bin/env python3
"""
Find ideal ROC/ROC of ROC periods based on SPY.csv data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_analyze_roc_data(csv_file_path):
    """Load and analyze ROC data to find ideal periods"""
    
    print("🔍 Loading and analyzing ROC data...")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Check if ROC columns exist
    if 'roc' not in df.columns or 'roc_of_roc' not in df.columns:
        print("❌ Error: ROC or ROC_of_ROC columns not found in data")
        return None
    
    # Remove rows with NaN in ROC columns
    df = df.dropna(subset=['roc', 'roc_of_roc'])
    
    print(f"📊 Data loaded: {len(df)} rows with ROC data")
    print(f"ROC range: {df['roc'].min()}-{df['roc'].max()}")
    print(f"ROC_of_ROC range: {df['roc_of_roc'].min()}-{df['roc_of_roc'].max()}")
    print()
    
    return df

def find_best_roc_combinations(df, metric='win_rate', top_n=10):
    """Find the best ROC/ROC_of_ROC combinations for a given metric"""
    
    print(f"🏆 Finding best ROC combinations for {metric}...")
    print("-" * 50)
    
    # Group by ROC and ROC_of_ROC and calculate mean performance
    grouped = df.groupby(['roc', 'roc_of_roc'])[metric].agg(['mean', 'count']).reset_index()
    grouped.columns = ['roc', 'roc_of_roc', f'{metric}_mean', 'count']
    
    # Sort by the metric (descending for win_rate, average_trade_profit, total_trades)
    if metric in ['win_rate', 'average_trade_profit', 'total_trades']:
        grouped = grouped.sort_values(f'{metric}_mean', ascending=False)
    else:
        grouped = grouped.sort_values(f'{metric}_mean', ascending=True)
    
    # Get top N combinations
    top_combinations = grouped.head(top_n)
    
    print(f"Top {top_n} ROC/ROC_of_ROC combinations for {metric}:")
    print()
    
    for i, row in top_combinations.iterrows():
        print(f"{i+1:2d}. ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f} → {metric}: {row[f'{metric}_mean']:.4f} (n={row['count']})")
    
    print()
    return top_combinations

def create_roc_performance_heatmap(df, metric='win_rate'):
    """Create a heatmap showing ROC performance"""
    
    print(f"📈 Creating {metric} heatmap...")
    
    # Create pivot table
    pivot = df.pivot_table(values=metric, index='roc_of_roc', columns='roc', aggfunc='mean')
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(title=metric.replace('_', ' ').title())
    ))
    
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} by ROC vs ROC_of_ROC Periods',
        xaxis_title='ROC Period',
        yaxis_title='ROC of ROC Period',
        width=800,
        height=600
    )
    
    fig.show()
    
    # Find the best combination from the heatmap
    best_idx = np.unravel_index(np.argmax(pivot.values), pivot.values.shape)
    best_roc_of_roc = pivot.index[best_idx[0]]
    best_roc = pivot.columns[best_idx[1]]
    best_value = pivot.values[best_idx]
    
    print(f"🔥 Best combination from heatmap: ROC={best_roc}, ROC_of_ROC={best_roc_of_roc} → {metric}: {best_value:.4f}")
    print()
    
    return best_roc, best_roc_of_roc, best_value

def analyze_roc_trade_volume(df):
    """Analyze ROC combinations by trade volume"""
    
    print("📊 Analyzing ROC combinations by trade volume...")
    print("-" * 50)
    
    # Group by ROC combinations and count trades
    grouped = df.groupby(['roc', 'roc_of_roc'])['total_trades'].agg(['sum', 'mean', 'count']).reset_index()
    grouped.columns = ['roc', 'roc_of_roc', 'total_trades_sum', 'avg_trades_per_config', 'num_configurations']
    
    # Sort by total trades
    grouped = grouped.sort_values('total_trades_sum', ascending=False)
    
    print("Top 10 ROC combinations by total trade volume:")
    print()
    
    for i, row in grouped.head(10).iterrows():
        print(f"{i+1:2d}. ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f} → Total trades: {row['total_trades_sum']:3.0f} (avg: {row['avg_trades_per_config']:.1f}, configs: {row['num_configurations']})")
    
    print()
    return grouped

def find_balanced_combinations(df):
    """Find ROC combinations with good balance of win rate and profit"""
    
    print("⚖️ Finding balanced ROC combinations (good win rate + profit)...")
    print("-" * 50)
    
    # Group by ROC combinations
    grouped = df.groupby(['roc', 'roc_of_roc']).agg({
        'win_rate': 'mean',
        'average_trade_profit': 'mean',
        'total_trades': 'sum',
        'max_drawdown': 'mean'
    }).reset_index()
    
    # Create a composite score (win_rate * average_trade_profit * log(total_trades))
    grouped['composite_score'] = (
        grouped['win_rate'] * 
        grouped['average_trade_profit'] * 
        np.log1p(grouped['total_trades'])
    )
    
    # Sort by composite score
    grouped = grouped.sort_values('composite_score', ascending=False)
    
    print("Top 10 balanced ROC combinations:")
    print()
    
    for i, row in grouped.head(10).iterrows():
        print(f"{i+1:2d}. ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f}")
        print(f"    Win Rate: {row['win_rate']:.3f}, Avg Profit: ${row['average_trade_profit']:.4f}")
        print(f"    Total Trades: {row['total_trades']:.0f}, Max Drawdown: {row['max_drawdown']:.4f}")
        print(f"    Composite Score: {row['composite_score']:.4f}")
        print()
    
    return grouped

def create_comprehensive_analysis(df):
    """Create comprehensive ROC analysis"""
    
    print("🎯 COMPREHENSIVE ROC/ROC_of_ROC ANALYSIS")
    print("=" * 60)
    
    # 1. Best win rate combinations
    print("\n1️⃣ BEST WIN RATE COMBINATIONS:")
    win_rate_best = find_best_roc_combinations(df, 'win_rate', 5)
    
    # 2. Best profit combinations
    print("\n2️⃣ BEST PROFIT COMBINATIONS:")
    profit_best = find_best_roc_combinations(df, 'average_trade_profit', 5)
    
    # 3. Most active combinations
    print("\n3️⃣ MOST ACTIVE COMBINATIONS:")
    volume_analysis = analyze_roc_trade_volume(df)
    
    # 4. Balanced combinations
    print("\n4️⃣ BALANCED COMBINATIONS:")
    balanced = find_balanced_combinations(df)
    
    # 5. Create heatmaps
    print("\n5️⃣ PERFORMANCE HEATMAPS:")
    create_roc_performance_heatmap(df, 'win_rate')
    create_roc_performance_heatmap(df, 'average_trade_profit')
    create_roc_performance_heatmap(df, 'total_trades')
    
    # 6. Summary recommendations
    print("\n6️⃣ SUMMARY RECOMMENDATIONS:")
    print("-" * 40)
    
    # Best overall (balanced)
    best_overall = balanced.iloc[0]
    print(f"🏆 BEST OVERALL: ROC={best_overall['roc']:.0f}, ROC_of_ROC={best_overall['roc_of_roc']:.0f}")
    print(f"   Win Rate: {best_overall['win_rate']:.3f}")
    print(f"   Avg Profit: ${best_overall['average_trade_profit']:.4f}")
    print(f"   Total Trades: {best_overall['total_trades']:.0f}")
    
    # Best win rate
    best_win_rate = win_rate_best.iloc[0]
    print(f"\n🎯 BEST WIN RATE: ROC={best_win_rate['roc']:.0f}, ROC_of_ROC={best_win_rate['roc_of_roc']:.0f}")
    print(f"   Win Rate: {best_win_rate['win_rate_mean']:.3f}")
    
    # Best profit
    best_profit = profit_best.iloc[0]
    print(f"\n💰 BEST PROFIT: ROC={best_profit['roc']:.0f}, ROC_of_ROC={best_profit['roc_of_roc']:.0f}")
    print(f"   Avg Profit: ${best_profit['average_trade_profit_mean']:.4f}")
    
    # Most active
    most_active = volume_analysis.iloc[0]
    print(f"\n📈 MOST ACTIVE: ROC={most_active['roc']:.0f}, ROC_of_ROC={most_active['roc_of_roc']:.0f}")
    print(f"   Total Trades: {most_active['total_trades_sum']:.0f}")
    
    return {
        'best_overall': best_overall,
        'best_win_rate': best_win_rate,
        'best_profit': best_profit,
        'most_active': most_active,
        'balanced_data': balanced,
        'win_rate_data': win_rate_best,
        'profit_data': profit_best,
        'volume_data': volume_analysis
    }

def main():
    """Main analysis function"""
    
    csv_file = "data/analysis/SPY.csv"
    
    try:
        # Load data
        df = load_and_analyze_roc_data(csv_file)
        
        if df is None:
            return
        
        # Run comprehensive analysis
        results = create_comprehensive_analysis(df)
        
        print("\n✅ Analysis complete! Check the heatmaps above for visual insights.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 