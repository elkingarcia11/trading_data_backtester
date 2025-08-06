#!/usr/bin/env python3
"""
Analyze the actual data patterns to understand the exact initialization methods being used.
"""

import pandas as pd
import numpy as np

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

def analyze_ema_pattern():
    """Analyze EMA initialization pattern"""
    print("ANALYZING EMA PATTERN")
    print("=" * 50)
    
    # Load test file
    test_file = "SPY250806C00634000.csv"
    actual_file = f'data/options_actual/{test_file}'
    options_file = f'data/options/{test_file}'
    
    actual_data = load_csv_with_fallback_encoding(actual_file)
    options_data = load_csv_with_fallback_encoding(options_file)
    
    if actual_data is None or options_data is None:
        print("Failed to load data files")
        return
    
    # Sort by timestamp
    actual_data = actual_data.sort_values('timestamp').reset_index(drop=True)
    options_data = options_data.sort_values('timestamp').reset_index(drop=True)
    
    # Find first valid EMA index
    ema_first_valid = actual_data['ema'].first_valid_index()
    print(f"First valid EMA index: {ema_first_valid}")
    
    # Check if it's SMA initialized
    if ema_first_valid is not None:
        # Check if the first EMA value equals SMA of preceding values
        ema_first_val = actual_data['ema'].iloc[ema_first_valid]
        print(f"First EMA value: {ema_first_val}")
        
        # Try different SMA periods
        for period in [8, 9, 10, 15, 20]:
            if ema_first_valid >= period - 1:
                sma_val = options_data['last_price'].iloc[ema_first_valid - period + 1:ema_first_valid + 1].mean()
                print(f"SMA({period}) ending at index {ema_first_valid}: {sma_val}")
                if abs(ema_first_val - sma_val) < 0.001:
                    print(f"*** MATCH! EMA appears to be initialized with SMA({period})")
        
        # Try EMA with different methods starting from different indices
        print(f"\nTesting pandas EMA methods:")
        for start_idx in [0, 5, 8, 9]:
            if start_idx <= ema_first_valid:
                price_subset = options_data['last_price'].iloc[start_idx:]
                ema_adjust_true = price_subset.ewm(span=20, adjust=True).mean()
                ema_adjust_false = price_subset.ewm(span=20, adjust=False).mean()
                
                offset_idx = ema_first_valid - start_idx
                if offset_idx < len(ema_adjust_true):
                    print(f"  Start at {start_idx}, adjust=True at {ema_first_valid}: {ema_adjust_true.iloc[offset_idx]}")
                    print(f"  Start at {start_idx}, adjust=False at {ema_first_valid}: {ema_adjust_false.iloc[offset_idx]}")

def analyze_vwma_pattern():
    """Analyze VWMA initialization pattern"""
    print("\nANALYZING VWMA PATTERN")
    print("=" * 50)
    
    # Load test file
    test_file = "SPY250806C00634000.csv"
    actual_file = f'data/options_actual/{test_file}'
    options_file = f'data/options/{test_file}'
    
    actual_data = load_csv_with_fallback_encoding(actual_file)
    options_data = load_csv_with_fallback_encoding(options_file)
    
    if actual_data is None or options_data is None:
        print("Failed to load data files")
        return
    
    # Sort by timestamp
    actual_data = actual_data.sort_values('timestamp').reset_index(drop=True)
    options_data = options_data.sort_values('timestamp').reset_index(drop=True)
    
    # Find first valid VWMA index
    vwma_first_valid = actual_data['vwma'].first_valid_index()
    print(f"First valid VWMA index: {vwma_first_valid}")
    
    if vwma_first_valid is not None:
        vwma_first_val = actual_data['vwma'].iloc[vwma_first_valid]
        print(f"First VWMA value: {vwma_first_val}")
        
        # Try different VWMA periods
        for period in [10, 12, 13, 15, 20]:
            if vwma_first_valid >= period - 1:
                start_idx = vwma_first_valid - period + 1
                price_window = options_data['last_price'].iloc[start_idx:vwma_first_valid + 1]
                volume_window = options_data['volume'].iloc[start_idx:vwma_first_valid + 1]
                
                numerator = (price_window * volume_window).sum()
                denominator = volume_window.sum()
                vwma_calc = numerator / denominator if denominator > 0 else np.nan
                
                print(f"VWMA({period}) ending at index {vwma_first_valid}: {vwma_calc}")
                if abs(vwma_first_val - vwma_calc) < 0.001:
                    print(f"*** MATCH! VWMA appears to use period {period}")

def analyze_roc_pattern():
    """Analyze ROC initialization pattern"""
    print("\nANALYZING ROC PATTERN")
    print("=" * 50)
    
    # Load test file
    test_file = "SPY250806C00634000.csv"
    actual_file = f'data/options_actual/{test_file}'
    options_file = f'data/options/{test_file}'
    
    actual_data = load_csv_with_fallback_encoding(actual_file)
    options_data = load_csv_with_fallback_encoding(options_file)
    
    if actual_data is None or options_data is None:
        print("Failed to load data files")
        return
    
    # Sort by timestamp
    actual_data = actual_data.sort_values('timestamp').reset_index(drop=True)
    options_data = options_data.sort_values('timestamp').reset_index(drop=True)
    
    # Find first valid ROC index
    roc_first_valid = actual_data['roc'].first_valid_index()
    print(f"First valid ROC index: {roc_first_valid}")
    
    if roc_first_valid is not None:
        roc_first_val = actual_data['roc'].iloc[roc_first_valid]
        print(f"First ROC value: {roc_first_val}")
        
        # Check what price values are being compared
        current_price = options_data['last_price'].iloc[roc_first_valid]
        print(f"Current price at index {roc_first_valid}: {current_price}")
        
        # Try different lookback periods and starting points
        for period in [9, 10, 11]:
            if roc_first_valid >= period:
                past_price = options_data['last_price'].iloc[roc_first_valid - period]
                roc_calc = ((current_price - past_price) / past_price) * 100
                print(f"ROC with period {period} (vs index {roc_first_valid - period}): {roc_calc}, past_price: {past_price}")
                if abs(roc_first_val - roc_calc) < 0.001:
                    print(f"*** MATCH! ROC appears to use period {period}")

def main():
    """Main analysis function"""
    print("Starting Actual Data Pattern Analysis")
    print("=" * 50)
    
    analyze_ema_pattern()
    analyze_vwma_pattern() 
    analyze_roc_pattern()

if __name__ == "__main__":
    main()