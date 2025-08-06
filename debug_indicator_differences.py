#!/usr/bin/env python3
"""
Debug script to investigate specific differences in EMA, ROC, and VWMA calculations
between our calculator and the actual pre-calculated data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the indicator-calculator directory to the path
sys.path.append('indicator-calculator')
from indicator_calculator import IndicatorCalculator

def load_csv_with_fallback_encoding(filepath):
    """Load CSV with fallback encoding handling"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            break
    
    return None

def debug_ema_calculation(data, actual_data, filename):
    """Debug EMA calculation differences"""
    print(f"\n{'='*60}")
    print(f"DEBUGGING EMA for {filename}")
    print(f"{'='*60}")
    
    # Calculate our EMA
    our_ema = IndicatorCalculator.calculate_ema(data, period=20, price_column='last_price')
    actual_ema = actual_data['ema']
    
    # Show first non-null values
    our_first_valid = our_ema.first_valid_index()
    actual_first_valid = actual_ema.first_valid_index()
    
    print(f"Our EMA first valid index: {our_first_valid}")
    print(f"Actual EMA first valid index: {actual_first_valid}")
    
    if our_first_valid is not None and actual_first_valid is not None:
        start_idx = max(our_first_valid, actual_first_valid)
        end_idx = min(start_idx + 10, len(data))
        
        comparison_df = pd.DataFrame({
            'timestamp': data['timestamp'].iloc[start_idx:end_idx],
            'last_price': data['last_price'].iloc[start_idx:end_idx],
            'our_ema': our_ema.iloc[start_idx:end_idx],
            'actual_ema': actual_ema.iloc[start_idx:end_idx],
            'difference': (our_ema - actual_ema).iloc[start_idx:end_idx]
        })
        
        print("\nFirst 10 comparable values:")
        print(comparison_df.to_string(index=False))
        
        # Try different EMA initialization methods
        print(f"\nTrying different EMA methods:")
        
        # Method 1: adjust=True (default pandas)
        ema_adjust_true = data['last_price'].ewm(span=20, adjust=True).mean()
        print(f"EMA with adjust=True at index {start_idx}: {ema_adjust_true.iloc[start_idx]:.10f}")
        
        # Method 2: adjust=False (our current method)
        ema_adjust_false = data['last_price'].ewm(span=20, adjust=False).mean()
        print(f"EMA with adjust=False at index {start_idx}: {ema_adjust_false.iloc[start_idx]:.10f}")
        
        # Method 3: Traditional EMA with different alpha
        alpha = 2 / (20 + 1)
        ema_traditional = data['last_price'].ewm(alpha=alpha, adjust=False).mean()
        print(f"EMA with alpha={alpha:.4f} at index {start_idx}: {ema_traditional.iloc[start_idx]:.10f}")
        
        # Method 4: Manual EMA calculation with SMA initialization
        manual_ema = pd.Series(index=data.index, dtype=float)
        sma_period = 20
        sma_values = data['last_price'].rolling(window=sma_period).mean()
        
        for i in range(len(data)):
            if i < sma_period - 1:
                manual_ema.iloc[i] = np.nan
            elif i == sma_period - 1:
                manual_ema.iloc[i] = sma_values.iloc[i]  # Initialize with SMA
            else:
                manual_ema.iloc[i] = alpha * data['last_price'].iloc[i] + (1 - alpha) * manual_ema.iloc[i-1]
        
        print(f"Manual EMA (SMA init) at index {start_idx}: {manual_ema.iloc[start_idx]:.10f}")
        print(f"Actual EMA at index {start_idx}: {actual_ema.iloc[start_idx]:.10f}")

def debug_roc_calculation(data, actual_data, filename):
    """Debug ROC calculation differences"""
    print(f"\n{'='*60}")
    print(f"DEBUGGING ROC for {filename}")
    print(f"{'='*60}")
    
    # Calculate our ROC
    our_roc = IndicatorCalculator.calculate_roc(data, period=10, price_column='last_price')
    actual_roc = actual_data['roc']
    
    # Show first non-null values
    our_first_valid = our_roc.first_valid_index()
    actual_first_valid = actual_roc.first_valid_index()
    
    print(f"Our ROC first valid index: {our_first_valid}")
    print(f"Actual ROC first valid index: {actual_first_valid}")
    
    if our_first_valid is not None and actual_first_valid is not None:
        start_idx = max(our_first_valid, actual_first_valid)
        end_idx = min(start_idx + 10, len(data))
        
        # Show the calculation step by step
        print(f"\nROC calculation breakdown (period=10):")
        for i in range(start_idx, min(start_idx + 5, len(data))):
            current_price = data['last_price'].iloc[i]
            prev_price = data['last_price'].iloc[i-10] if i >= 10 else np.nan
            our_calc = ((current_price - prev_price) / prev_price) * 100 if not pd.isna(prev_price) else np.nan
            
            print(f"Index {i}: current={current_price:.6f}, prev={prev_price:.6f}, "
                  f"our_calc={our_calc:.6f}, actual={actual_roc.iloc[i]:.6f}")

def debug_vwma_calculation(data, actual_data, filename):
    """Debug VWMA calculation differences"""
    print(f"\n{'='*60}")
    print(f"DEBUGGING VWMA for {filename}")
    print(f"{'='*60}")
    
    # Calculate our VWMA
    our_vwma = IndicatorCalculator.calculate_vwma(data, period=20, is_options=True)
    actual_vwma = actual_data['vwma']
    
    # Show first non-null values
    our_first_valid = our_vwma.first_valid_index()
    actual_first_valid = actual_vwma.first_valid_index()
    
    print(f"Our VWMA first valid index: {our_first_valid}")
    print(f"Actual VWMA first valid index: {actual_first_valid}")
    
    if our_first_valid is not None and actual_first_valid is not None:
        start_idx = max(our_first_valid, actual_first_valid)
        end_idx = min(start_idx + 5, len(data))
        
        # Show the calculation step by step
        print(f"\nVWMA calculation breakdown (period=20):")
        for i in range(start_idx, end_idx):
            if i >= 19:  # Ensure we have enough data for period=20
                window_start = i - 19
                window_prices = data['last_price'].iloc[window_start:i+1]
                window_volumes = data['volume'].iloc[window_start:i+1]
                
                our_numerator = (window_prices * window_volumes).sum()
                our_denominator = window_volumes.sum()
                our_calc = our_numerator / our_denominator if our_denominator != 0 else np.nan
                
                print(f"Index {i}: numerator={our_numerator:.2f}, denominator={our_denominator:.0f}, "
                      f"our_calc={our_calc:.6f}, actual={actual_vwma.iloc[i]:.6f}")
                
                # Show sample data
                print(f"  Sample prices: {window_prices.tail(3).tolist()}")
                print(f"  Sample volumes: {window_volumes.tail(3).tolist()}")

def main():
    """Main debugging function"""
    print("Starting Indicator Calculation Debugging")
    print("=" * 50)
    
    # Load configuration
    import indicator_config
    config = {}
    for key, value in indicator_config.CUSTOM_CONFIG.items():
        if hasattr(value, '__iter__') and not isinstance(value, (str, int)):
            config[key] = list(value)[0]
        else:
            config[key] = value
    
    print(f"Using configuration: {config}")
    
    # Test with one file first
    test_file = "SPY250806C00634000.csv"
    
    options_file = Path('data/options') / test_file
    actual_file = Path('data/options_actual') / test_file
    
    print(f"Debugging file: {test_file}")
    
    # Load data
    data = load_csv_with_fallback_encoding(str(options_file))
    actual_data = load_csv_with_fallback_encoding(str(actual_file))
    
    if data is None or actual_data is None:
        print("Failed to load data files")
        return
    
    # Sort by timestamp
    data = data.sort_values('timestamp').reset_index(drop=True)
    actual_data = actual_data.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Data shape: {data.shape}")
    print(f"Actual data shape: {actual_data.shape}")
    
    # Check columns
    print(f"Data columns: {data.columns.tolist()}")
    print(f"Available indicator columns in actual: {[col for col in actual_data.columns if col in ['ema', 'vwma', 'roc', 'roc_of_roc']]}")
    
    # Debug each indicator
    debug_ema_calculation(data, actual_data, test_file)
    debug_roc_calculation(data, actual_data, test_file)
    debug_vwma_calculation(data, actual_data, test_file)

if __name__ == "__main__":
    main()