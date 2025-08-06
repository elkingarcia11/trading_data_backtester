#!/usr/bin/env python3
"""
Analyze EMA/VWMA configurations for optimal performance
Focusing on: Low max drawdown, High win rate, High average profit
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_filter_data(csv_file_path):
    """Load data and filter for EMA/VWMA analysis"""
    
    print("🔍 Loading and filtering data for EMA/VWMA analysis...")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Check if required columns exist
    required_cols = ['ema', 'vwma', 'total_trades', 'win_rate', 'average_trade_profit', 'max_drawdown']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        return None
    
    # Remove rows with NaN in required columns
    df = df.dropna(subset=required_cols)
    
    print(f"📊 Total data rows: {len(df)}")
    print(f"📊 EMA range: {df['ema'].min()}-{df['ema'].max()}")
    print(f"📊 VWMA range: {df['vwma'].min()}-{df['vwma'].max()}")
    print()
    
    return df

def find_best_combinations_by_metric(df, metric, top_n=10, ascending=False):
    """Find best EMA/VWMA combinations for a specific metric"""
    
    print(f"🏆 Finding best EMA/VWMA combinations for {metric}...")
    print("-" * 50)
    
    # Group by EMA and VWMA and calculate mean performance
    grouped = df.groupby(['ema', 'vwma'])[metric].agg(['mean', 'count']).reset_index()
    grouped.columns = ['ema', 'vwma', f'{metric}_mean', 'count']
    
    # Sort by the metric
    grouped = grouped.sort_values(f'{metric}_mean', ascending=ascending)
    
    # Get top N combinations
    top_combinations = grouped.head(top_n)
    
    print(f"Top {top_n} EMA/VWMA combinations for {metric}:")
    print()
    
    for i, row in top_combinations.iterrows():
        print(f"{i+1:2d}. EMA={row['ema']:2.0f}, VWMA={row['vwma']:2.0f} → {metric}: {row[f'{metric}_mean']:.4f} (n={row['count']:.0f})")
    
    print()
    return top_combinations

def find_optimal_combinations(df):
    """Find optimal combinations considering all three criteria"""
    
    print("🎯 Finding optimal EMA/VWMA combinations (low drawdown, high win rate, high profit)...")
    print("-" * 70)
    
    # Group by EMA/VWMA combinations
    grouped = df.groupby(['ema', 'vwma']).agg({
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
    
    print("Top 15 optimal EMA/VWMA combinations:")
    print()
    
    for i, row in grouped.head(15).iterrows():
        print(f"{i+1:2d}. EMA={row['ema']:2.0f}, VWMA={row['vwma']:2.0f}")
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
        pivot = df.pivot_table(values=metric, index='vwma', columns='ema', aggfunc='mean', fill_value=np.nan)
        
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
            title=f'{name} by EMA vs VWMA',
            xaxis_title='EMA Period',
            yaxis_title='VWMA Period',
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
            
            best_vwma = pivot.index[best_idx[0]]
            best_ema = pivot.columns[best_idx[1]]
            best_value = pivot.values[best_idx]
            
            if not np.isnan(best_value):
                print(f"🔥 Best {name}: EMA={best_ema}, VWMA={best_vwma} → {best_value:.4f}")
            else:
                print(f"⚠️ No valid data found for {name}")
        else:
            print(f"⚠️ No valid data found for {name}")
        print()

def create_trade_duration_heatmaps(df):
    """Create trade duration heatmaps for EMA/VWMA combinations"""
    
    print("📊 Creating trade duration heatmaps...")
    print("-" * 40)
    
    # Check if we have the required columns
    required_cols = ['ema', 'vwma', 'win_rate', 'average_trade_profit', 'max_drawdown', 'average_trade_duration (minutes)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        print("Available columns:", list(df.columns))
        return
    
    # Create duration bins for better visualization
    df['duration_bin'] = pd.cut(df['average_trade_duration (minutes)'], bins=5, labels=['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow'])
    
    # Create pivot tables for each metric by duration
    metrics = {
        'Win Rate': 'win_rate',
        'Average Profit': 'average_trade_profit', 
        'Max Drawdown': 'max_drawdown'
    }
    
    for metric_name, metric_col in metrics.items():
        print(f"Creating {metric_name} by Trade Duration heatmap...")
        
        try:
            # Create pivot table with duration bins
            pivot = df.pivot_table(
                values=metric_col,
                index=['vwma', 'duration_bin'],
                columns='ema',
                aggfunc='mean',
                fill_value=np.nan
            )
            
            if pivot.empty or pivot.isna().all().all():
                print(f"⚠️ No valid data found for {metric_name} by duration")
                continue
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=[f"{idx[0]}-{idx[1]}" for idx in pivot.index],
                colorscale='RdYlBu_r' if metric_col != 'max_drawdown' else 'Reds',
                showscale=True,
                colorbar=dict(title=metric_name),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f'{metric_name} by EMA vs VWMA + Trade Duration',
                xaxis_title='EMA Period',
                yaxis_title='VWMA Period + Duration Bin',
                width=1000,
                height=800
            )
            
            fig.show()
            
        except Exception as e:
            print(f"❌ Error creating {metric_name} duration heatmap: {e}")
            continue
    
    # Also create a summary heatmap showing best duration for each EMA/VWMA combination
    print("Creating Best Duration Summary heatmap...")
    
    try:
        # Find the best duration bin for each EMA/VWMA combination
        best_duration = df.groupby(['ema', 'vwma']).apply(
            lambda x: x.loc[x['win_rate'].idxmax(), 'duration_bin'] if len(x) > 0 else None
        ).reset_index()
        best_duration.columns = ['ema', 'vwma', 'best_duration_bin']
        
        # Create pivot table for best duration
        duration_pivot = best_duration.pivot_table(
            values='best_duration_bin',
            index='vwma',
            columns='ema',
            aggfunc='first'
        )
        
        if not duration_pivot.empty:
            fig = go.Figure(data=go.Heatmap(
                z=duration_pivot.values,
                x=duration_pivot.columns,
                y=duration_pivot.index,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title='Best Duration'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Best Trade Duration for EMA/VWMA Combinations',
                xaxis_title='EMA Period',
                yaxis_title='VWMA Period',
                width=800,
                height=600
            )
            
            fig.show()
        
    except Exception as e:
        print(f"❌ Error creating duration summary heatmap: {e}")

def analyze_trade_distribution(df):
    """Analyze distribution of trades across EMA/VWMA combinations"""
    
    print("📊 Analyzing trade distribution...")
    print("-" * 40)
    
    # Group by EMA/VWMA combinations and analyze trade counts
    trade_analysis = df.groupby(['ema', 'vwma']).agg({
        'total_trades': ['mean', 'min', 'max', 'count']
    }).reset_index()
    
    trade_analysis.columns = ['ema', 'vwma', 'avg_trades', 'min_trades', 'max_trades', 'config_count']
    
    # Find combinations with reasonable trade counts (10-100)
    reasonable_trades = trade_analysis[
        (trade_analysis['avg_trades'] >= 10) & 
        (trade_analysis['avg_trades'] <= 100)
    ].sort_values('avg_trades', ascending=False)
    
    print("EMA/VWMA combinations with reasonable trade counts (10-100):")
    print()
    
    for i, row in reasonable_trades.head(10).iterrows():
        print(f"{i+1:2d}. EMA={row['ema']:2.0f}, VWMA={row['vwma']:2.0f}")
        print(f"    Avg Trades: {row['avg_trades']:.1f} (range: {row['min_trades']:.0f}-{row['max_trades']:.0f})")
        print(f"    Configurations: {row['config_count']:.0f}")
        print()
    
    return reasonable_trades

def analyze_individual_indicators(df):
    """Analyze EMA and VWMA performance individually"""
    
    print("📊 Analyzing individual indicator performance...")
    print("-" * 50)
    
    # Analyze EMA performance
    print("EMA Performance Analysis:")
    ema_performance = df.groupby('ema').agg({
        'win_rate': 'mean',
        'average_trade_profit': 'mean',
        'max_drawdown': 'mean',
        'total_trades': 'mean'
    }).reset_index()
    
    # Find best EMA periods
    best_ema_win = ema_performance.loc[ema_performance['win_rate'].idxmax()]
    best_ema_profit = ema_performance.loc[ema_performance['average_trade_profit'].idxmax()]
    best_ema_drawdown = ema_performance.loc[ema_performance['max_drawdown'].idxmax()]  # idxmax for drawdown (closest to 0)
    
    print(f"  Best Win Rate: EMA={best_ema_win['ema']:.0f} → {best_ema_win['win_rate']:.3f}")
    print(f"  Best Profit: EMA={best_ema_profit['ema']:.0f} → ${best_ema_profit['average_trade_profit']:.4f}")
    print(f"  Lowest Drawdown: EMA={best_ema_drawdown['ema']:.0f} → {best_ema_drawdown['max_drawdown']:.4f}")
    print()
    
    # Analyze VWMA performance
    print("VWMA Performance Analysis:")
    vwma_performance = df.groupby('vwma').agg({
        'win_rate': 'mean',
        'average_trade_profit': 'mean',
        'max_drawdown': 'mean',
        'total_trades': 'mean'
    }).reset_index()
    
    # Find best VWMA periods
    best_vwma_win = vwma_performance.loc[vwma_performance['win_rate'].idxmax()]
    best_vwma_profit = vwma_performance.loc[vwma_performance['average_trade_profit'].idxmax()]
    best_vwma_drawdown = vwma_performance.loc[vwma_performance['max_drawdown'].idxmax()]  # idxmax for drawdown (closest to 0)
    
    print(f"  Best Win Rate: VWMA={best_vwma_win['vwma']:.0f} → {best_vwma_win['win_rate']:.3f}")
    print(f"  Best Profit: VWMA={best_vwma_profit['vwma']:.0f} → ${best_vwma_profit['average_trade_profit']:.4f}")
    print(f"  Lowest Drawdown: VWMA={best_vwma_drawdown['vwma']:.0f} → {best_vwma_drawdown['max_drawdown']:.4f}")
    print()
    
    return ema_performance, vwma_performance

def create_comprehensive_analysis(df):
    """Run comprehensive analysis for EMA/VWMA"""
    
    print("🎯 COMPREHENSIVE EMA/VWMA ANALYSIS")
    print("=" * 60)
    
    # 1. Individual indicator analysis
    print("\n1️⃣ INDIVIDUAL INDICATOR ANALYSIS:")
    ema_perf, vwma_perf = analyze_individual_indicators(df)
    
    # 2. Best win rate combinations
    print("\n2️⃣ BEST WIN RATE COMBINATIONS:")
    win_rate_best = find_best_combinations_by_metric(df, 'win_rate', 5, ascending=False)
    
    # 3. Best profit combinations
    print("\n3️⃣ BEST PROFIT COMBINATIONS:")
    profit_best = find_best_combinations_by_metric(df, 'average_trade_profit', 5, ascending=False)
    
    # 4. Lowest drawdown combinations (closest to 0)
    print("\n4️⃣ LOWEST DRAWDOWN COMBINATIONS (closest to 0):")
    drawdown_best = find_best_combinations_by_metric(df, 'max_drawdown', 5, ascending=False)
    
    # 5. Optimal combinations (balanced)
    print("\n5️⃣ OPTIMAL COMBINATIONS (BALANCED):")
    optimal = find_optimal_combinations(df)
    
    # 6. Trade distribution analysis
    print("\n6️⃣ TRADE DISTRIBUTION ANALYSIS:")
    trade_dist = analyze_trade_distribution(df)
    
    # 7. Create heatmaps
    print("\n7️⃣ PERFORMANCE HEATMAPS:")
    create_performance_heatmaps(df)
    
    # 8. Create trade duration heatmaps
    print("\n8️⃣ TRADE DURATION HEATMAPS:")
    create_trade_duration_heatmaps(df)
    
    # 9. Summary recommendations
    print("\n9️⃣ SUMMARY RECOMMENDATIONS:")
    print("-" * 40)
    
    # Best overall (optimal)
    best_overall = optimal.iloc[0]
    print(f"🏆 BEST OVERALL: EMA={best_overall['ema']:.0f}, VWMA={best_overall['vwma']:.0f}")
    print(f"   Win Rate: {best_overall['win_rate']:.3f}")
    print(f"   Avg Profit: ${best_overall['average_trade_profit']:.4f}")
    print(f"   Max Drawdown: {best_overall['max_drawdown']:.4f}")
    print(f"   Avg Trades: {best_overall['total_trades']:.1f}")
    
    # Best win rate
    best_win_rate = win_rate_best.iloc[0]
    print(f"\n🎯 BEST WIN RATE: EMA={best_win_rate['ema']:.0f}, VWMA={best_win_rate['vwma']:.0f}")
    print(f"   Win Rate: {best_win_rate['win_rate_mean']:.3f}")
    
    # Best profit
    best_profit = profit_best.iloc[0]
    print(f"\n💰 BEST PROFIT: EMA={best_profit['ema']:.0f}, VWMA={best_profit['vwma']:.0f}")
    print(f"   Avg Profit: ${best_profit['average_trade_profit_mean']:.4f}")
    
    # Lowest drawdown (closest to 0)
    best_drawdown = drawdown_best.iloc[0]
    print(f"\n🛡️ LOWEST DRAWDOWN (closest to 0): EMA={best_drawdown['ema']:.0f}, VWMA={best_drawdown['vwma']:.0f}")
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
        'trade_distribution': trade_dist,
        'ema_performance': ema_perf,
        'vwma_performance': vwma_perf
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
        results = create_comprehensive_analysis(df)
        
        print("\n✅ Analysis complete! Check the heatmaps above for visual insights.")
        print("\n💡 Key Insights:")
        print("   • EMA and VWMA are moving average indicators that can work well together")
        print("   • Shorter periods tend to be more responsive but may generate more noise")
        print("   • Longer periods provide smoother signals but may lag behind price action")
        print("   • Balance between responsiveness and stability is crucial")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 