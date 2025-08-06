#!/usr/bin/env python3
"""
Check trade distribution in the data
"""

import pandas as pd
import numpy as np

def analyze_trade_distribution(csv_file_path):
    """Analyze the distribution of trade counts in the data"""
    
    print("🔍 Analyzing trade distribution in data...")
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
    
    print(f"📊 Total data rows: {len(df)}")
    print()
    
    # Analyze trade count distribution
    trade_counts = df['total_trades'].value_counts().sort_index()
    
    print("📊 Trade Count Distribution:")
    print("-" * 40)
    for count, frequency in trade_counts.items():
        print(f"  {count:2d} trades: {frequency:4d} configurations")
    
    print()
    
    # Show statistics
    print("📊 Trade Count Statistics:")
    print("-" * 40)
    print(f"  Minimum trades: {df['total_trades'].min()}")
    print(f"  Maximum trades: {df['total_trades'].max()}")
    print(f"  Mean trades: {df['total_trades'].mean():.2f}")
    print(f"  Median trades: {df['total_trades'].median():.2f}")
    print(f"  Standard deviation: {df['total_trades'].std():.2f}")
    
    print()
    
    # Check different ranges
    ranges_to_check = [
        (1, 5, "1-5 trades"),
        (3, 10, "3-10 trades"),
        (5, 15, "5-15 trades"),
        (10, 20, "10-20 trades"),
        (1, 20, "1-20 trades")
    ]
    
    print("📊 Configurations by Trade Range:")
    print("-" * 40)
    for min_trades, max_trades, label in ranges_to_check:
        count = len(df[(df['total_trades'] >= min_trades) & (df['total_trades'] <= max_trades)])
        print(f"  {label}: {count:4d} configurations")
    
    print()
    
    # Show top performers by different metrics
    print("🏆 Top 5 Configurations by Win Rate:")
    print("-" * 40)
    top_win_rate = df.nlargest(5, 'win_rate')[['roc', 'roc_of_roc', 'total_trades', 'win_rate', 'average_trade_profit', 'max_drawdown']]
    for i, row in top_win_rate.iterrows():
        print(f"  ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f} → {row['total_trades']:2.0f} trades, {row['win_rate']:.3f} win rate")
    
    print()
    
    print("💰 Top 5 Configurations by Average Profit:")
    print("-" * 40)
    top_profit = df.nlargest(5, 'average_trade_profit')[['roc', 'roc_of_roc', 'total_trades', 'win_rate', 'average_trade_profit', 'max_drawdown']]
    for i, row in top_profit.iterrows():
        print(f"  ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f} → {row['total_trades']:2.0f} trades, ${row['average_trade_profit']:.4f} profit")
    
    print()
    
    print("🛡️ Top 5 Configurations by Lowest Drawdown:")
    print("-" * 40)
    top_drawdown = df.nlargest(5, 'max_drawdown')[['roc', 'roc_of_roc', 'total_trades', 'win_rate', 'average_trade_profit', 'max_drawdown']]
    for i, row in top_drawdown.iterrows():
        print(f"  ROC={row['roc']:2.0f}, ROC_of_ROC={row['roc_of_roc']:2.0f} → {row['total_trades']:2.0f} trades, {row['max_drawdown']:.4f} drawdown")
    
    return df

def main():
    """Main function"""
    
    csv_file = "data/analysis/SPY.csv"
    
    try:
        df = analyze_trade_distribution(csv_file)
        
        if df is not None:
            print("\n💡 Recommendations:")
            print("   • Check what trade count range is most common in your data")
            print("   • Consider adjusting the analysis range based on actual data distribution")
            print("   • Focus on configurations that generate reasonable trade counts for your strategy")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 