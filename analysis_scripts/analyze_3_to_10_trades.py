#!/usr/bin/env python3
"""
Analyze ROC/ROC of ROC periods for 3-10 trades configurations
Focusing on: Low max drawdown, High win rate, High average profit
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_filter_data(csv_file_path):
    """Load data and filter for 3-10 trades"""
    
    print("🔍 Loading and filtering data for 3-10 trades...")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Check if required columns exist
    required_cols = ['roc', 'roc_of_roc', 'total_trades', 'win_rate', 'average_trade_profit', 'max_drawdown']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        return None
    
    # Remove rows with NaN in required columns
    df = df.dropna(subset=required_cols)
    
    # Filter for 3-10 trades
    df_filtered = df[(df['total_trades'] >= 3) & (df['total_trades'] <= 10)].copy()
    
    print(f"📊 Original data: {len(df)} rows")
    print(f"📊 Filtered data (3-10 trades): {len(df_filtered)} rows")
    
    if len(df_filtered) == 0:
        print("⚠️ No configurations found with 3-10 trades!")
        print("💡 Available trade ranges in your data:")
        trade_counts = df['total_trades'].value_counts().sort_index()
        print("   Trade counts available:")
        for count, freq in trade_counts.head(20).items():
            print(f"   {count:2d} trades: {freq:4d} configurations")
        if len(trade_counts) > 20:
            print(f"   ... and {len(trade_counts) - 20} more trade counts")
        print()
        print("💡 Suggestions:")
        print("   • Try a different trade range (e.g., 10-30 trades)")
        print("   • Adjust your backtesting parameters to generate fewer trades")
        print("   • Use a different time period or symbol")
        return None
    
    print(f"ROC range: {df_filtered['roc'].min()}-{df_filtered['roc'].max()}")
    print(f"ROC_of_ROC range: {df_filtered['roc_of_roc'].min()}-{df_filtered['roc_of_roc'].max()}")
    
    # Show unique ROC combinations
    unique_combinations = df_filtered[['roc', 'roc_of_roc']].drop_duplicates()
    print(f"📊 Unique ROC combinations: {len(unique_combinations)}")
    print(f"📊 ROC values: {sorted(df_filtered['roc'].unique())}")
    print(f"📊 ROC_of_ROC values: {sorted(df_filtered['roc_of_roc'].unique())}")
    print()
    
    return df_filtered

def find_best_combinations_by_metric(df, metric, top_n=10, ascending=False):
    """Find best ROC combinations for a specific metric"""
    
    print(f"🏆 Finding best ROC combinations for {metric}...")
    print("-" * 50)
    
    # Group by ROC and ROC_of_ROC and calculate mean performance
    grouped = df.groupby(['roc', 'roc_of_roc'])[metric].agg(['mean', 'count']).reset_index()
    grouped.columns = ['roc', 'roc_of_roc', f'{metric}_mean', 'count']
    
    # Sort by the metric
    grouped = grouped.sort_values(f'{metric}_mean', ascending=ascending)
    
    # Get top N combinations
    top_combinations = grouped.head(top_n)
    
    print(f"Top {top_n} ROC/ROC_of_ROC combinations for {metric}:")
    print()
    
    for i, row in top_combinations.iterrows():
        print(f"{i+1:2d}. ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f} → {metric}: {row[f'{metric}_mean']:.4f} (n={row['count']:.0f})")
    
    print()
    return top_combinations

def find_optimal_combinations(df):
    """Find optimal combinations considering all three criteria"""
    
    print("🎯 Finding optimal ROC combinations (low drawdown, high win rate, high profit)...")
    print("-" * 70)
    
    # Group by ROC combinations
    grouped = df.groupby(['roc', 'roc_of_roc']).agg({
        'win_rate': 'mean',
        'average_trade_profit': 'mean',
        'max_drawdown': 'mean',
        'total_trades': 'mean',
        'total_trade_profit': 'sum'
    }).reset_index()
    
    # Create composite score
    # Higher is better: win_rate + average_trade_profit - abs(max_drawdown)
    # Normalize max_drawdown to positive scale (lower drawdown = higher score)
    max_drawdown_abs = abs(grouped['max_drawdown'])
    max_drawdown_normalized = max_drawdown_abs / max_drawdown_abs.max()
    
    grouped['composite_score'] = (
        grouped['win_rate'] * 0.4 +  # 40% weight to win rate
        grouped['average_trade_profit'] * 100 +  # 40% weight to profit (scaled up)
        (1 - max_drawdown_normalized) * 0.2  # 20% weight to low drawdown
    )
    
    # Sort by composite score
    grouped = grouped.sort_values('composite_score', ascending=False)
    
    print("Top 15 optimal ROC combinations:")
    print()
    
    for i, row in grouped.head(15).iterrows():
        print(f"{i+1:2d}. ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f}")
        print(f"    Win Rate: {row['win_rate']:.3f}, Avg Profit: ${row['average_trade_profit']:.4f}")
        print(f"    Max Drawdown: {row['max_drawdown']:.4f}, Avg Trades: {row['total_trades']:.1f}")
        print(f"    Total Profit: ${row['total_trade_profit']:.4f}, Score: {row['composite_score']:.4f}")
        print()
    
    return grouped

def create_performance_heatmaps(df):
    """Create heatmaps for key metrics"""
    
    print("📈 Creating performance heatmaps...")
    print("-" * 40)
    
    metrics = ['win_rate', 'average_trade_profit', 'max_drawdown']
    metric_names = ['Win Rate', 'Average Profit', 'Max Drawdown']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        print(f"Creating {name} heatmap...")
        
        # Create pivot table with fill_value to handle missing combinations
        pivot = df.pivot_table(values=metric, index='roc_of_roc', columns='roc', aggfunc='mean', fill_value=np.nan)
        
        # Check if we have valid data
        if pivot.empty or pivot.isna().all().all():
            print(f"⚠️ No data available for {name} heatmap")
            print()
            continue
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlBu_r' if metric != 'max_drawdown' else 'Reds',
            showscale=True,
            colorbar=dict(title=name),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'{name} by ROC vs ROC_of_ROC (3-10 trades)',
            xaxis_title='ROC Period',
            yaxis_title='ROC of ROC Period',
            width=800,
            height=600
        )
        
        fig.show()
        
        # Find best combination (ignore NaN values)
        valid_mask = ~np.isnan(pivot.values)
        if valid_mask.any():
            if metric == 'max_drawdown':
                # For drawdown, we want the value closest to 0 (least negative)
                best_idx = np.unravel_index(np.argmax(pivot.values[valid_mask]), pivot.values.shape)
            else:
                best_idx = np.unravel_index(np.argmax(pivot.values[valid_mask]), pivot.values.shape)
            
            best_roc_of_roc = pivot.index[best_idx[0]]
            best_roc = pivot.columns[best_idx[1]]
            best_value = pivot.values[best_idx]
            
            if not np.isnan(best_value):
                print(f"🔥 Best {name}: ROC={best_roc}, ROC_of_ROC={best_roc_of_roc} → {best_value:.4f}")
            else:
                print(f"⚠️ No valid data found for {name}")
        else:
            print(f"⚠️ No valid data found for {name}")
        print()

def analyze_trade_distribution(df):
    """Analyze distribution of trades across ROC combinations"""
    
    print("📊 Analyzing trade distribution...")
    print("-" * 40)
    
    # Group by ROC combinations and analyze trade counts
    trade_analysis = df.groupby(['roc', 'roc_of_roc']).agg({
        'total_trades': ['mean', 'min', 'max', 'count']
    }).reset_index()
    
    trade_analysis.columns = ['roc', 'roc_of_roc', 'avg_trades', 'min_trades', 'max_trades', 'config_count']
    
    # Find combinations with consistent 3-10 trades
    consistent_trades = trade_analysis[
        (trade_analysis['min_trades'] >= 3) & 
        (trade_analysis['max_trades'] <= 10)
    ].sort_values('avg_trades', ascending=False)
    
    print("ROC combinations with consistent 3-10 trades:")
    print()
    
    for i, row in consistent_trades.head(10).iterrows():
        print(f"{i+1:2d}. ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f}")
        print(f"    Avg Trades: {row['avg_trades']:.1f} (range: {row['min_trades']:.0f}-{row['max_trades']:.0f})")
        print(f"    Configurations: {row['config_count']:.0f}")
        print()
    
    return consistent_trades

def create_comprehensive_analysis(df):
    """Run comprehensive analysis for 3-10 trades"""
    
    print("🎯 COMPREHENSIVE ANALYSIS: 3-10 TRADES")
    print("=" * 60)
    
    # 1. Best win rate combinations
    print("\n1️⃣ BEST WIN RATE COMBINATIONS:")
    win_rate_best = find_best_combinations_by_metric(df, 'win_rate', 5, ascending=False)
    
    # 2. Best profit combinations
    print("\n2️⃣ BEST PROFIT COMBINATIONS:")
    profit_best = find_best_combinations_by_metric(df, 'average_trade_profit', 5, ascending=False)
    
    # 3. Lowest drawdown combinations (closest to 0)
    print("\n3️⃣ LOWEST DRAWDOWN COMBINATIONS (closest to 0):")
    drawdown_best = find_best_combinations_by_metric(df, 'max_drawdown', 5, ascending=False)
    
    # 4. Optimal combinations (balanced)
    print("\n4️⃣ OPTIMAL COMBINATIONS (BALANCED):")
    optimal = find_optimal_combinations(df)
    
    # 5. Trade distribution analysis
    print("\n5️⃣ TRADE DISTRIBUTION ANALYSIS:")
    trade_dist = analyze_trade_distribution(df)
    
    # 6. Create heatmaps
    print("\n6️⃣ PERFORMANCE HEATMAPS:")
    create_performance_heatmaps(df)
    
    # 7. Summary recommendations
    print("\n7️⃣ SUMMARY RECOMMENDATIONS:")
    print("-" * 40)
    
    # Best overall (optimal)
    best_overall = optimal.iloc[0]
    print(f"🏆 BEST OVERALL: ROC={best_overall['roc']:.0f}, ROC_of_ROC={best_overall['roc_of_roc']:.0f}")
    print(f"   Win Rate: {best_overall['win_rate']:.3f}")
    print(f"   Avg Profit: ${best_overall['average_trade_profit']:.4f}")
    print(f"   Max Drawdown: {best_overall['max_drawdown']:.4f}")
    print(f"   Avg Trades: {best_overall['total_trades']:.1f}")
    
    # Best win rate
    best_win_rate = win_rate_best.iloc[0]
    print(f"\n🎯 BEST WIN RATE: ROC={best_win_rate['roc']:.0f}, ROC_of_ROC={best_win_rate['roc_of_roc']:.0f}")
    print(f"   Win Rate: {best_win_rate['win_rate_mean']:.3f}")
    
    # Best profit
    best_profit = profit_best.iloc[0]
    print(f"\n💰 BEST PROFIT: ROC={best_profit['roc']:.0f}, ROC_of_ROC={best_profit['roc_of_roc']:.0f}")
    print(f"   Avg Profit: ${best_profit['average_trade_profit_mean']:.4f}")
    
    # Lowest drawdown (closest to 0)
    best_drawdown = drawdown_best.iloc[0]
    print(f"\n🛡️ LOWEST DRAWDOWN (closest to 0): ROC={best_drawdown['roc']:.0f}, ROC_of_ROC={best_drawdown['roc_of_roc']:.0f}")
    print(f"   Max Drawdown: {best_drawdown['max_drawdown_mean']:.4f}")
    
    return {
        'best_overall': best_overall,
        'best_win_rate': best_win_rate,
        'best_profit': best_profit,
        'best_drawdown': best_drawdown,
        'optimal_data': optimal,
        'win_rate_data': win_rate_best,
        'profit_data': profit_best,
        'drawdown_data': drawdown_best,
        'trade_distribution': trade_dist
    }

def main():
    """Main analysis function"""
    
    csv_file = "data/analysis/SPY.csv"
    
    try:
        # Load and filter data
        df = load_and_filter_data(csv_file)
        
        if df is None:
            return
        
        # Run comprehensive analysis
        if df is not None:
            results = create_comprehensive_analysis(df)
        else:
            print("❌ Cannot proceed with analysis - no data available for 3-10 trades range.")
            return
        
        print("\n✅ Analysis complete! Check the heatmaps above for visual insights.")
        print("\n💡 Key Insights:")
        print("   • Focus on configurations with 3-10 trades for better risk management")
        print("   • Lower ROC periods tend to generate more trades")
        print("   • Higher ROC periods may provide better win rates but fewer opportunities")
        print("   • Balance between win rate, profit, and drawdown is crucial")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 