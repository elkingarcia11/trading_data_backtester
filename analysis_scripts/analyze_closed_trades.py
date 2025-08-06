import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_closed_trades_data(csv_file_path):
    """Load and prepare closed trades data for analysis"""
    
    print("🔍 Loading closed trades data...")
    print("=" * 50)
    
    try:
        df = pd.read_csv(csv_file_path)
        print(f"📊 Total trades loaded: {len(df)}")
        print(f"📊 Date range: {df['entry_timestamp'].min()} to {df['entry_timestamp'].max()}")
        print(f"📊 Symbols: {df['symbol'].unique()}")
        print(f"📊 Contract types: {df['contract_type'].unique()}")
        
        # Convert timestamps to datetime
        df['entry_timestamp'] = pd.to_datetime(df['entry_timestamp'])
        df['exit_timestamp'] = pd.to_datetime(df['exit_timestamp'])
        
        # Basic statistics
        print(f"📊 Average profit %: {df['profit_pct'].mean():.2f}%")
        print(f"📊 Profit % range: {df['profit_pct'].min():.2f}% to {df['profit_pct'].max():.2f}%")
        print(f"📊 Average duration: {df['duration_minutes'].mean():.2f} minutes")
        print(f"📊 Win rate: {(df['profit_pct'] > 0).mean():.2%}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_profit_heatmaps(df):
    """Create heatmaps showing profit percentage by various factors"""
    
    print("📈 Creating profit percentage heatmaps...")
    print("-" * 40)
    
    # 1. Profit % by Entry EMA vs Entry VWMA
    print("Creating Profit % by Entry EMA vs Entry VWMA heatmap...")
    
    try:
        # Show the actual ranges for each bin
        print("\n📊 EMA Value Ranges:")
        ema_bins = pd.cut(df['entry_ema'], bins=5, retbins=True)
        ema_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        for i, (label, start, end) in enumerate(zip(ema_labels, ema_bins[1][:-1], ema_bins[1][1:])):
            print(f"   {label}: {start:.3f} to {end:.3f}")
        
        print("\n📊 VWMA Value Ranges:")
        vwma_bins = pd.cut(df['entry_vwma'], bins=5, retbins=True)
        vwma_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        for i, (label, start, end) in enumerate(zip(vwma_labels, vwma_bins[1][:-1], vwma_bins[1][1:])):
            print(f"   {label}: {start:.3f} to {end:.3f}")
        
        # Create bins for EMA and VWMA for better visualization
        df['ema_bin'] = pd.cut(df['entry_ema'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        df['vwma_bin'] = pd.cut(df['entry_vwma'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Create pivot table
        pivot = df.pivot_table(
            values='profit_pct',
            index='vwma_bin',
            columns='ema_bin',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        if not pivot.empty and not pivot.isna().all().all():
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Profit %'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Average Profit % by Entry EMA vs Entry VWMA',
                xaxis_title='Entry EMA Level',
                yaxis_title='Entry VWMA Level',
                width=800,
                height=600
            )
            
            fig.show()
        else:
            print("⚠️ No valid data for EMA vs VWMA heatmap")
            
    except Exception as e:
        print(f"❌ Error creating EMA vs VWMA heatmap: {e}")
    
    # 2. Profit % by Duration vs Profit %
    print("Creating Profit % by Duration heatmap...")
    
    try:
        # Show the actual ranges for duration bins
        print("\n📊 Duration Ranges:")
        duration_bins = pd.cut(df['duration_minutes'], bins=5, retbins=True)
        duration_labels = ['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow']
        for i, (label, start, end) in enumerate(zip(duration_labels, duration_bins[1][:-1], duration_bins[1][1:])):
            print(f"   {label}: {start:.2f} to {end:.2f} minutes")
        
        # Create duration bins
        df['duration_bin'] = pd.cut(df['duration_minutes'], bins=5, labels=['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow'])
        
        # Create pivot table
        duration_pivot = df.pivot_table(
            values='profit_pct',
            index='duration_bin',
            columns='symbol',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        if not duration_pivot.empty and not duration_pivot.isna().all().all():
            fig = go.Figure(data=go.Heatmap(
                z=duration_pivot.values,
                x=duration_pivot.columns,
                y=duration_pivot.index,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Profit %'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Average Profit % by Trade Duration vs Symbol',
                xaxis_title='Symbol',
                yaxis_title='Trade Duration',
                width=800,
                height=600
            )
            
            fig.show()
        else:
            print("⚠️ No valid data for duration heatmap")
            
    except Exception as e:
        print(f"❌ Error creating duration heatmap: {e}")
    
    # 3. Profit % by MACD Signal vs Stochastic
    print("Creating Profit % by MACD vs Stochastic heatmap...")
    
    try:
        # Show the actual ranges for MACD and Stochastic bins
        print("\n📊 MACD Line Ranges:")
        macd_bins = pd.cut(df['entry_macd_line'], bins=5, retbins=True)
        macd_labels = ['Very Bearish', 'Bearish', 'Neutral', 'Bullish', 'Very Bullish']
        for i, (label, start, end) in enumerate(zip(macd_labels, macd_bins[1][:-1], macd_bins[1][1:])):
            print(f"   {label}: {start:.4f} to {end:.4f}")
        
        print("\n📊 Stochastic K Ranges:")
        stoch_bins = pd.cut(df['entry_stoch_k'], bins=5, retbins=True)
        stoch_labels = ['Oversold', 'Low', 'Medium', 'High', 'Overbought']
        for i, (label, start, end) in enumerate(zip(stoch_labels, stoch_bins[1][:-1], stoch_bins[1][1:])):
            print(f"   {label}: {start:.1f} to {end:.1f}")
        
        # Create bins for MACD and Stochastic
        df['macd_bin'] = pd.cut(df['entry_macd_line'], bins=5, labels=['Very Bearish', 'Bearish', 'Neutral', 'Bullish', 'Very Bullish'])
        df['stoch_bin'] = pd.cut(df['entry_stoch_k'], bins=5, labels=['Oversold', 'Low', 'Medium', 'High', 'Overbought'])
        
        # Create pivot table
        macd_stoch_pivot = df.pivot_table(
            values='profit_pct',
            index='stoch_bin',
            columns='macd_bin',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        if not macd_stoch_pivot.empty and not macd_stoch_pivot.isna().all().all():
            fig = go.Figure(data=go.Heatmap(
                z=macd_stoch_pivot.values,
                x=macd_stoch_pivot.columns,
                y=macd_stoch_pivot.index,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Profit %'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Average Profit % by Entry MACD vs Entry Stochastic',
                xaxis_title='Entry MACD Signal',
                yaxis_title='Entry Stochastic Level',
                width=800,
                height=600
            )
            
            fig.show()
        else:
            print("⚠️ No valid data for MACD vs Stochastic heatmap")
            
    except Exception as e:
        print(f"❌ Error creating MACD vs Stochastic heatmap: {e}")

def create_comprehensive_analysis(df):
    """Create comprehensive analysis of closed trades"""
    
    print("🎯 COMPREHENSIVE CLOSED TRADES ANALYSIS")
    print("=" * 50)
    
    # 1. Overall Statistics
    print("\n1️⃣ OVERALL STATISTICS:")
    print("-" * 30)
    
    total_trades = len(df)
    winning_trades = len(df[df['profit_pct'] > 0])
    losing_trades = len(df[df['profit_pct'] < 0])
    breakeven_trades = len(df[df['profit_pct'] == 0])
    
    print(f"📊 Total Trades: {total_trades}")
    print(f"📊 Winning Trades: {winning_trades} ({winning_trades/total_trades:.1%})")
    print(f"📊 Losing Trades: {losing_trades} ({losing_trades/total_trades:.1%})")
    print(f"📊 Breakeven Trades: {breakeven_trades} ({breakeven_trades/total_trades:.1%})")
    print(f"📊 Average Profit %: {df['profit_pct'].mean():.2f}%")
    print(f"📊 Median Profit %: {df['profit_pct'].median():.2f}%")
    print(f"📊 Best Trade: {df['profit_pct'].max():.2f}%")
    print(f"📊 Worst Trade: {df['profit_pct'].min():.2f}%")
    print(f"📊 Average Duration: {df['duration_minutes'].mean():.2f} minutes")
    
    # 2. Symbol Analysis
    print("\n2️⃣ SYMBOL ANALYSIS:")
    print("-" * 30)
    
    symbol_stats = df.groupby('symbol').agg({
        'profit_pct': ['count', 'mean', 'median', 'min', 'max'],
        'duration_minutes': 'mean'
    }).round(2)
    
    symbol_stats.columns = ['Trades', 'Avg_Profit_%', 'Median_Profit_%', 'Min_Profit_%', 'Max_Profit_%', 'Avg_Duration_Min']
    print(symbol_stats)
    
    # 3. Contract Type Analysis
    print("\n3️⃣ CONTRACT TYPE ANALYSIS:")
    print("-" * 30)
    
    contract_stats = df.groupby('contract_type').agg({
        'profit_pct': ['count', 'mean', 'median', 'min', 'max'],
        'duration_minutes': 'mean'
    }).round(2)
    
    contract_stats.columns = ['Trades', 'Avg_Profit_%', 'Median_Profit_%', 'Min_Profit_%', 'Max_Profit_%', 'Avg_Duration_Min']
    print(contract_stats)
    
    # 4. Duration Analysis
    print("\n4️⃣ DURATION ANALYSIS:")
    print("-" * 30)
    
    # Create duration bins
    df['duration_bin'] = pd.cut(df['duration_minutes'], bins=5, labels=['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow'])
    
    duration_stats = df.groupby('duration_bin').agg({
        'profit_pct': ['count', 'mean', 'median', 'min', 'max'],
        'duration_minutes': 'mean'
    }).round(2)
    
    duration_stats.columns = ['Trades', 'Avg_Profit_%', 'Median_Profit_%', 'Min_Profit_%', 'Max_Profit_%', 'Avg_Duration_Min']
    print(duration_stats)
    
    # 5. Exit Reason Analysis
    print("\n5️⃣ EXIT REASON ANALYSIS:")
    print("-" * 30)
    
    exit_stats = df.groupby('exit_reason').agg({
        'profit_pct': ['count', 'mean', 'median', 'min', 'max'],
        'duration_minutes': 'mean'
    }).round(2)
    
    exit_stats.columns = ['Trades', 'Avg_Profit_%', 'Median_Profit_%', 'Min_Profit_%', 'Max_Profit_%', 'Avg_Duration_Min']
    print(exit_stats)
    
    # 6. Best Performing Combinations
    print("\n6️⃣ BEST PERFORMING COMBINATIONS:")
    print("-" * 30)
    
    # Find best combinations by profit %
    best_profit = df.nlargest(10, 'profit_pct')[['symbol', 'contract_type', 'profit_pct', 'duration_minutes', 'entry_ema', 'entry_vwma']]
    print("Top 10 Most Profitable Trades:")
    print(best_profit.round(2))
    
    # 7. Worst Performing Combinations
    print("\n7️⃣ WORST PERFORMING COMBINATIONS:")
    print("-" * 30)
    
    worst_profit = df.nsmallest(10, 'profit_pct')[['symbol', 'contract_type', 'profit_pct', 'duration_minutes', 'entry_ema', 'entry_vwma']]
    print("Top 10 Least Profitable Trades:")
    print(worst_profit.round(2))

def create_advanced_heatmaps(df):
    """Create advanced heatmaps for deeper analysis"""
    
    print("\n8️⃣ ADVANCED HEATMAPS:")
    print("-" * 30)
    
    # 1. Profit % by Entry vs Exit Indicator Changes
    print("Creating Profit % by Indicator Changes heatmap...")
    
    try:
        # Calculate indicator changes
        df['ema_change'] = df['exit_ema'] - df['entry_ema']
        df['vwma_change'] = df['exit_vwma'] - df['entry_vwma']
        df['macd_change'] = df['exit_macd_line'] - df['entry_macd_line']
        df['stoch_change'] = df['exit_stoch_k'] - df['entry_stoch_k']
        
        # Create bins for changes
        df['ema_change_bin'] = pd.cut(df['ema_change'], bins=5, labels=['Large Decrease', 'Decrease', 'No Change', 'Increase', 'Large Increase'])
        df['vwma_change_bin'] = pd.cut(df['vwma_change'], bins=5, labels=['Large Decrease', 'Decrease', 'No Change', 'Increase', 'Large Increase'])
        
        # Create pivot table
        change_pivot = df.pivot_table(
            values='profit_pct',
            index='vwma_change_bin',
            columns='ema_change_bin',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        if not change_pivot.empty and not change_pivot.isna().all().all():
            fig = go.Figure(data=go.Heatmap(
                z=change_pivot.values,
                x=change_pivot.columns,
                y=change_pivot.index,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Profit %'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Average Profit % by EMA vs VWMA Changes During Trade',
                xaxis_title='EMA Change During Trade',
                yaxis_title='VWMA Change During Trade',
                width=800,
                height=600
            )
            
            fig.show()
        else:
            print("⚠️ No valid data for indicator changes heatmap")
            
    except Exception as e:
        print(f"❌ Error creating indicator changes heatmap: {e}")
    
    # 2. Profit % by Strike Price vs Duration
    print("Creating Profit % by Strike Price vs Duration heatmap...")
    
    try:
        # Create strike price bins
        df['strike_bin'] = pd.cut(df['strike_price'], bins=5, labels=['Low Strike', 'Medium-Low', 'Medium', 'Medium-High', 'High Strike'])
        
        # Create pivot table
        strike_duration_pivot = df.pivot_table(
            values='profit_pct',
            index='duration_bin',
            columns='strike_bin',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        if not strike_duration_pivot.empty and not strike_duration_pivot.isna().all().all():
            fig = go.Figure(data=go.Heatmap(
                z=strike_duration_pivot.values,
                x=strike_duration_pivot.columns,
                y=strike_duration_pivot.index,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Profit %'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Average Profit % by Strike Price vs Trade Duration',
                xaxis_title='Strike Price Level',
                yaxis_title='Trade Duration',
                width=800,
                height=600
            )
            
            fig.show()
        else:
            print("⚠️ No valid data for strike price vs duration heatmap")
            
    except Exception as e:
        print(f"❌ Error creating strike price vs duration heatmap: {e}")

def create_indicator_difference_heatmaps(df):
    """Create heatmaps analyzing differences between indicator pairs"""
    
    print("\n9️⃣ INDICATOR DIFFERENCE HEATMAPS:")
    print("-" * 30)
    
    # Filter for trades where EMA > VWMA at entry (your entry condition)
    df_filtered = df[df['entry_ema'] > df['entry_vwma']].copy()
    print(f"📊 Trades with EMA > VWMA at entry: {len(df_filtered)} out of {len(df)} total trades")
    
    if len(df_filtered) == 0:
        print("⚠️ No trades found with EMA > VWMA at entry")
        return
    
    # 1. EMA vs VWMA Difference Analysis (EMA above VWMA only)
    print("Creating EMA vs VWMA Difference heatmaps (EMA > VWMA only)...")
    
    try:
        # Calculate differences at entry and exit
        df_filtered['entry_ema_vwma_diff'] = df_filtered['entry_ema'] - df_filtered['entry_vwma']
        df_filtered['exit_ema_vwma_diff'] = df_filtered['exit_ema'] - df_filtered['exit_vwma']
        df_filtered['ema_vwma_diff_change'] = df_filtered['exit_ema_vwma_diff'] - df_filtered['entry_ema_vwma_diff']
        
        # Show ranges for differences
        print("\n📊 EMA vs VWMA Difference Ranges (EMA > VWMA):")
        diff_bins = pd.cut(df_filtered['entry_ema_vwma_diff'], bins=5, retbins=True)
        diff_labels = ['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above']
        for i, (label, start, end) in enumerate(zip(diff_labels, diff_bins[1][:-1], diff_bins[1][1:])):
            print(f"   {label}: {start:.4f} to {end:.4f}")
        
        # Create bins for differences
        df_filtered['entry_ema_vwma_diff_bin'] = pd.cut(df_filtered['entry_ema_vwma_diff'], bins=5, labels=['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above'])
        df_filtered['exit_ema_vwma_diff_bin'] = pd.cut(df_filtered['exit_ema_vwma_diff'], bins=5, labels=['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above'])
        
        # Create pivot table for entry differences
        entry_diff_pivot = df_filtered.pivot_table(
            values='profit_pct',
            index='entry_ema_vwma_diff_bin',
            columns='symbol',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        if not entry_diff_pivot.empty and not entry_diff_pivot.isna().all().all():
            fig = go.Figure(data=go.Heatmap(
                z=entry_diff_pivot.values,
                x=entry_diff_pivot.columns,
                y=entry_diff_pivot.index,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Profit %'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Average Profit % by How Much EMA is Above VWMA at Entry',
                xaxis_title='Symbol',
                yaxis_title='EMA Above VWMA Level at Entry',
                width=800,
                height=600
            )
            
            fig.show()
        else:
            print("⚠️ No valid data for entry EMA vs VWMA difference heatmap")
            
    except Exception as e:
        print(f"❌ Error creating EMA vs VWMA difference heatmap: {e}")
    
    # 2. MACD Line vs Signal Difference Analysis (Line above Signal only)
    print("Creating MACD Line vs Signal Difference heatmaps (Line > Signal only)...")
    
    try:
        # Filter for trades where MACD Line > Signal at entry
        df_macd_filtered = df_filtered[df_filtered['entry_macd_line'] > df_filtered['entry_macd_signal']].copy()
        print(f"📊 Trades with MACD Line > Signal at entry: {len(df_macd_filtered)} out of {len(df_filtered)} EMA>VWMA trades")
        
        if len(df_macd_filtered) > 0:
            # Calculate differences at entry and exit
            df_macd_filtered['entry_macd_diff'] = df_macd_filtered['entry_macd_line'] - df_macd_filtered['entry_macd_signal']
            df_macd_filtered['exit_macd_diff'] = df_macd_filtered['exit_macd_line'] - df_macd_filtered['exit_macd_signal']
            df_macd_filtered['macd_diff_change'] = df_macd_filtered['exit_macd_diff'] - df_macd_filtered['entry_macd_diff']
            
            # Show ranges for differences
            print("\n📊 MACD Line vs Signal Difference Ranges (Line > Signal):")
            macd_diff_bins = pd.cut(df_macd_filtered['entry_macd_diff'], bins=5, retbins=True)
            macd_diff_labels = ['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above']
            for i, (label, start, end) in enumerate(zip(macd_diff_labels, macd_diff_bins[1][:-1], macd_diff_bins[1][1:])):
                print(f"   {label}: {start:.4f} to {end:.4f}")
            
            # Create bins for differences
            df_macd_filtered['entry_macd_diff_bin'] = pd.cut(df_macd_filtered['entry_macd_diff'], bins=5, labels=['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above'])
            df_macd_filtered['exit_macd_diff_bin'] = pd.cut(df_macd_filtered['exit_macd_diff'], bins=5, labels=['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above'])
            
            # Create pivot table for entry differences
            entry_macd_pivot = df_macd_filtered.pivot_table(
                values='profit_pct',
                index='entry_macd_diff_bin',
                columns='symbol',
                aggfunc='mean',
                fill_value=np.nan
            )
            
            if not entry_macd_pivot.empty and not entry_macd_pivot.isna().all().all():
                fig = go.Figure(data=go.Heatmap(
                    z=entry_macd_pivot.values,
                    x=entry_macd_pivot.columns,
                    y=entry_macd_pivot.index,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title='Profit %'),
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Average Profit % by How Much MACD Line is Above Signal at Entry',
                    xaxis_title='Symbol',
                    yaxis_title='MACD Line Above Signal Level at Entry',
                    width=800,
                    height=600
                )
                
                fig.show()
            else:
                print("⚠️ No valid data for entry MACD difference heatmap")
        else:
            print("⚠️ No trades found with MACD Line > Signal at entry")
            
    except Exception as e:
        print(f"❌ Error creating MACD difference heatmap: {e}")
    
    # 3. Stochastic K vs D Difference Analysis (K above D only)
    print("Creating Stochastic K vs D Difference heatmaps (K > D only)...")
    
    try:
        # Filter for trades where Stochastic K > D at entry
        df_stoch_filtered = df_filtered[df_filtered['entry_stoch_k'] > df_filtered['entry_stoch_d']].copy()
        print(f"📊 Trades with Stochastic K > D at entry: {len(df_stoch_filtered)} out of {len(df_filtered)} EMA>VWMA trades")
        
        if len(df_stoch_filtered) > 0:
            # Calculate differences at entry and exit
            df_stoch_filtered['entry_stoch_diff'] = df_stoch_filtered['entry_stoch_k'] - df_stoch_filtered['entry_stoch_d']
            df_stoch_filtered['exit_stoch_diff'] = df_stoch_filtered['exit_stoch_k'] - df_stoch_filtered['exit_stoch_d']
            df_stoch_filtered['stoch_diff_change'] = df_stoch_filtered['exit_stoch_diff'] - df_stoch_filtered['entry_stoch_diff']
            
            # Show ranges for differences
            print("\n📊 Stochastic K vs D Difference Ranges (K > D):")
            stoch_diff_bins = pd.cut(df_stoch_filtered['entry_stoch_diff'], bins=5, retbins=True)
            stoch_diff_labels = ['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above']
            for i, (label, start, end) in enumerate(zip(stoch_diff_labels, stoch_diff_bins[1][:-1], stoch_diff_bins[1][1:])):
                print(f"   {label}: {start:.1f} to {end:.1f}")
            
            # Create bins for differences
            df_stoch_filtered['entry_stoch_diff_bin'] = pd.cut(df_stoch_filtered['entry_stoch_diff'], bins=5, labels=['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above'])
            df_stoch_filtered['exit_stoch_diff_bin'] = pd.cut(df_stoch_filtered['exit_stoch_diff'], bins=5, labels=['Slightly Above', 'Above', 'Well Above', 'Much Above', 'Very Much Above'])
            
            # Create pivot table for entry differences
            entry_stoch_pivot = df_stoch_filtered.pivot_table(
                values='profit_pct',
                index='entry_stoch_diff_bin',
                columns='symbol',
                aggfunc='mean',
                fill_value=np.nan
            )
            
            if not entry_stoch_pivot.empty and not entry_stoch_pivot.isna().all().all():
                fig = go.Figure(data=go.Heatmap(
                    z=entry_stoch_pivot.values,
                    x=entry_stoch_pivot.columns,
                    y=entry_stoch_pivot.index,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title='Profit %'),
                    hoverongaps=False
                ))
                
                fig.update_layout(
                    title='Average Profit % by How Much Stochastic K is Above D at Entry',
                    xaxis_title='Symbol',
                    yaxis_title='Stochastic K Above D Level at Entry',
                    width=800,
                    height=600
                )
                
                fig.show()
            else:
                print("⚠️ No valid data for entry Stochastic difference heatmap")
        else:
            print("⚠️ No trades found with Stochastic K > D at entry")
            
    except Exception as e:
        print(f"❌ Error creating Stochastic difference heatmap: {e}")
    
    # 4. Change During Trade Analysis
    print("Creating Change During Trade Analysis heatmaps...")
    
    try:
        # Use the main filtered dataset for change analysis
        # Create bins for changes during trade
        df_filtered['ema_vwma_change_bin'] = pd.cut(df_filtered['ema_vwma_diff_change'], bins=5, labels=['Large Decrease', 'Decrease', 'No Change', 'Increase', 'Large Increase'])
        
        # Calculate MACD and Stochastic changes for the main filtered dataset
        df_filtered['macd_diff_change'] = (df_filtered['exit_macd_line'] - df_filtered['exit_macd_signal']) - (df_filtered['entry_macd_line'] - df_filtered['entry_macd_signal'])
        df_filtered['stoch_diff_change'] = (df_filtered['exit_stoch_k'] - df_filtered['exit_stoch_d']) - (df_filtered['entry_stoch_k'] - df_filtered['entry_stoch_d'])
        
        df_filtered['macd_change_bin'] = pd.cut(df_filtered['macd_diff_change'], bins=5, labels=['Large Decrease', 'Decrease', 'No Change', 'Increase', 'Large Increase'])
        df_filtered['stoch_change_bin'] = pd.cut(df_filtered['stoch_diff_change'], bins=5, labels=['Large Decrease', 'Decrease', 'No Change', 'Increase', 'Large Increase'])
        
        # Show ranges for changes
        print("\n📊 Change During Trade Ranges:")
        print("EMA vs VWMA Difference Change:")
        change_bins = pd.cut(df_filtered['ema_vwma_diff_change'], bins=5, retbins=True)
        change_labels = ['Large Decrease', 'Decrease', 'No Change', 'Increase', 'Large Increase']
        for i, (label, start, end) in enumerate(zip(change_labels, change_bins[1][:-1], change_bins[1][1:])):
            print(f"   {label}: {start:.4f} to {end:.4f}")
        
        # Create pivot table for EMA/VWMA change vs profit
        change_pivot = df_filtered.pivot_table(
            values='profit_pct',
            index='ema_vwma_change_bin',
            columns='symbol',
            aggfunc='mean',
            fill_value=np.nan
        )
        
        if not change_pivot.empty and not change_pivot.isna().all().all():
            fig = go.Figure(data=go.Heatmap(
                z=change_pivot.values,
                x=change_pivot.columns,
                y=change_pivot.index,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Profit %'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Average Profit % by EMA vs VWMA Difference Change During Trade',
                xaxis_title='Symbol',
                yaxis_title='EMA vs VWMA Difference Change During Trade',
                width=800,
                height=600
            )
            
            fig.show()
        else:
            print("⚠️ No valid data for change during trade heatmap")
            
    except Exception as e:
        print(f"❌ Error creating change during trade heatmap: {e}")
    
    # 5. Summary Statistics
    print("\n📊 SUMMARY STATISTICS FOR FILTERED TRADES:")
    print("-" * 40)
    
    print(f"Total trades with EMA > VWMA: {len(df_filtered)}")
    print(f"Average profit %: {df_filtered['profit_pct'].mean():.2f}%")
    print(f"Win rate: {(df_filtered['profit_pct'] > 0).mean():.1%}")
    
    if len(df_macd_filtered) > 0:
        print(f"Trades with MACD Line > Signal: {len(df_macd_filtered)}")
        print(f"Average profit % (MACD filtered): {df_macd_filtered['profit_pct'].mean():.2f}%")
        print(f"Win rate (MACD filtered): {(df_macd_filtered['profit_pct'] > 0).mean():.1%}")
    
    if len(df_stoch_filtered) > 0:
        print(f"Trades with Stochastic K > D: {len(df_stoch_filtered)}")
        print(f"Average profit % (Stoch filtered): {df_stoch_filtered['profit_pct'].mean():.2f}%")
        print(f"Win rate (Stoch filtered): {(df_stoch_filtered['profit_pct'] > 0).mean():.1%}")

def main():
    """Main analysis function"""
    
    # Load data
    df = load_closed_trades_data('data/trades/close.csv')
    
    if df is None or df.empty:
        print("❌ No data to analyze")
        return
    
    # Run comprehensive analysis
    create_comprehensive_analysis(df)
    
    # Create heatmaps
    create_profit_heatmaps(df)
    
    # Create advanced heatmaps
    create_advanced_heatmaps(df)
    
    # Create indicator difference heatmaps
    create_indicator_difference_heatmaps(df)
    
    print("\n✅ Analysis complete! Check the heatmaps above for visual insights.")
    
    print("\n💡 Key Insights:")
    print("   • Analyze which indicator combinations lead to the highest profits")
    print("   • Identify optimal trade durations for different market conditions")
    print("   • Understand how indicator changes during trades affect profitability")
    print("   • Find patterns in strike price selection and timing")

if __name__ == "__main__":
    main()
