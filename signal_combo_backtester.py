#!/usr/bin/env python3
"""
Backtester that uses signal combinations to test different indicator combinations
"""

import pandas as pd
import sys
import os
from typing import List, Dict
import traceback
from generate_signal_combos import load_combinations_from_file
import random
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import itertools
sys.path.append('indicator-calculator')
from indicator_calculator import IndicatorCalculator

def validate_row_indicators(row: pd.Series, combo: Dict) -> bool:
    """
    Validate that all required indicators for a combination exist and are not NaN in the row.
    
    Args:
        row (pd.Series): Row with indicator values
        combo (Dict): Signal combination dictionary
        
    Returns:
        bool: True if all required indicators exist and are not NaN
    """
    required_indicators = []
    
    # Check which indicators are needed based on the combo
    for indicator in combo.keys():
        if indicator == 'stoch_rsi_k':
            required_indicators.extend(['stoch_rsi_k', 'stoch_rsi_d'])
        elif indicator == 'ema':
            required_indicators.extend(['ema', 'vwma'])
        elif indicator == 'macd_histogram':
            required_indicators.extend(['macd_line', 'macd_signal'])
        else:
            required_indicators.append(indicator)
    
    # Remove duplicates
    required_indicators = list(set(required_indicators))
    
    # Check if all required indicators exist in row and are not NaN
    for indicator in required_indicators:
        if indicator not in row or pd.isna(row[indicator]):
            return False
    
    return True

def evaluate_signal_combo(row: pd.Series, combo: Dict) -> bool:
    """
    Evaluate if buy/sell signal combinations are met.
    Handles ignored indicators (indicators with '_ignored' suffix).
    
    Args:
        row (pd.Series): DataFrame row with indicator values
        combo (Dict): Signal combination dictionary
        
    Returns:
        bool: True if signal conditions are met
    """
    condition_met = 0
    conditions_to_be_met = 0
    
    # Evaluate conditions using direct comparison instead of eval()
    for indicator, condition in combo.items():
        # If indicator is not in combo, it's ignored (no need to check for _ignored suffix)
            
        try:
            if indicator == 'roc' and 'roc' in row:
                try:
                    roc_val = float(row['roc'])
                    if condition == '>' and roc_val > 0:
                        condition_met += 1
                    elif condition == '<' and roc_val < 0:
                        condition_met += 1
                    elif condition == '>=' and roc_val >= 0:
                        condition_met += 1
                    elif condition == '<=' and roc_val <= 0:
                        condition_met += 1
                    conditions_to_be_met += 1
                except (ValueError, TypeError):
                    print(f"Error evaluating 'roc' with condition {condition}")
                    continue
                
            elif indicator == 'stoch_rsi_k' and 'stoch_rsi_k' in row and 'stoch_rsi_d' in row:
                try:
                    k_val = float(row['stoch_rsi_k'])
                    d_val = float(row['stoch_rsi_d'])
                    if condition == '>' and k_val > d_val:
                        condition_met += 1
                    elif condition == '<' and k_val < d_val:
                        condition_met += 1
                    elif condition == '>=' and k_val >= d_val:
                        condition_met += 1
                    elif condition == '<=' and k_val <= d_val:
                        condition_met += 1
                    conditions_to_be_met += 1
                except (ValueError, TypeError):
                    print(f"Error evaluating 'stoch_rsi_k' with condition {condition}")
                    continue
                
            elif indicator == 'macd_histogram' and 'macd_line' in row and 'macd_signal' in row:
                try:
                    macd_line_val = float(row['macd_line'])
                    macd_signal_val = float(row['macd_signal'])
                    # histogram > 0 means macd_line > macd_signal
                    # histogram < 0 means macd_line < macd_signal
                    if condition == '>' and macd_line_val > macd_signal_val:
                        condition_met += 1
                    elif condition == '<' and macd_line_val < macd_signal_val:
                        condition_met += 1
                    elif condition == '>=' and macd_line_val >= macd_signal_val:
                        condition_met += 1
                    elif condition == '<=' and macd_line_val <= macd_signal_val:
                        condition_met += 1
                    conditions_to_be_met += 1
                except (ValueError, TypeError):
                    print(f"Error evaluating 'macd_histogram' with condition {condition}")
                    # Skip this condition if we can't convert to float
                    continue
                
            elif indicator == 'macd_signal' and 'macd_signal' in row:
                try:
                    signal_val = float(row['macd_signal'])
                    if condition == '>' and signal_val > 0:
                        condition_met += 1
                    elif condition == '<' and signal_val < 0:
                        condition_met += 1
                    elif condition == '>=' and signal_val >= 0:
                        condition_met += 1
                    elif condition == '<=' and signal_val <= 0:
                        condition_met += 1
                    conditions_to_be_met += 1
                except (ValueError, TypeError):
                    print(f"Error evaluating 'macd_signal' with condition {condition}")
                    continue
                
            elif indicator == 'macd_line' and 'macd_line' in row:
                try:
                    line_val = float(row['macd_line'])
                    if condition == '>' and line_val > 0:
                        condition_met += 1
                    elif condition == '<' and line_val < 0:
                        condition_met += 1
                    elif condition == '>=' and line_val >= 0:
                        condition_met += 1
                    elif condition == '<=' and line_val <= 0:
                        condition_met += 1
                    conditions_to_be_met += 1
                except (ValueError, TypeError):
                    print(f"Error evaluating 'macd_line' with condition {condition}")
                    continue
                
            elif indicator == 'ema' and 'ema' in row and 'vwma' in row:
                try:
                    ema_val = float(row['ema'])
                    vwma_val = float(row['vwma'])
                    if condition == '>' and ema_val > vwma_val:
                        condition_met += 1
                    elif condition == '<' and ema_val < vwma_val:
                        condition_met += 1
                    elif condition == '>=' and ema_val >= vwma_val:
                        condition_met += 1
                    elif condition == '<=' and ema_val <= vwma_val:
                        condition_met += 1
                    conditions_to_be_met += 1
                except (ValueError, TypeError):
                    print(f"Error evaluating 'ema/vwma' with condition {condition}")
                    continue
                
        except (ValueError, TypeError, AttributeError):
            # Skip this condition if there's an error evaluating it
            continue

    # All conditions must be met for a signal
    return condition_met >= conditions_to_be_met and conditions_to_be_met > 0

def custom_backtest_with_combos(data: pd.DataFrame, buy_combo: Dict, sell_combo: Dict) -> Dict:
    """
    Run a custom backtest using separate buy/sell signal combinations.
    
    Args:
        data (pd.DataFrame): Price data with indicators
        buy_combo (Dict): Buy signal combination dictionary
        sell_combo (Dict): Sell signal combination dictionary
        
    Returns:
        Dict: Backtest results
    """
    # Make a copy to avoid modifying original data
    data = data.copy()
    
    trades = []
    trade_open = False
    entry_price = 0
    trade_start_timestamp = 0
    max_price_seen = 0
    min_price_seen = 0
    
    # Trailing stop and stop loss
    trailing_stop_pct = 0.9
    stop_loss_pct = 0.95
    valid_row = False
    for i in range(len(data)):
        row = data.iloc[i]
        try:
            current_price = float(row.get('last_price', 0))
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            if trade_open:
                sell_signal = evaluate_signal_combo(row, sell_combo)
                # Update unrealized profit/loss
                max_price_seen = max(max_price_seen, current_price)
                min_price_seen = min(min_price_seen, current_price)
                
                # Check exit conditions
                should_exit = (
                    sell_signal or
                    current_price <= entry_price * stop_loss_pct or
                    current_price <= max_price_seen * trailing_stop_pct
                )
                
                if should_exit:
                    # Close trade
                    # Use row index for duration if timestamps are strings
                    current_timestamp = row.get('timestamp', i)
                    if isinstance(current_timestamp, str) or isinstance(trade_start_timestamp, str):
                        trade_duration = i - (trade_start_timestamp if isinstance(trade_start_timestamp, int) else 0)
                    else:
                        trade_duration = current_timestamp - trade_start_timestamp
                    profit = current_price - entry_price
                    max_unrealized_profit = max_price_seen - entry_price
                    max_drawdown = min_price_seen - entry_price
                    
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'trade_duration': trade_duration,
                        'profit': profit,
                        'max_unrealized_profit': max_unrealized_profit,
                        'max_drawdown': max_drawdown
                    })
                    # Reset trade state
                    trade_open = False
                    entry_price = 0
                    trade_start_timestamp = 0
                    max_price_seen = 0
                    min_price_seen = 0
            else:
                # Check if all required indicators are present in this row for buy/sell signals
                if not valid_row:
                    valid_row = validate_row_indicators(row, buy_combo) and validate_row_indicators(row, sell_combo)
                    if not valid_row:
                        continue
                buy_signal = evaluate_signal_combo(row, buy_combo)
                # Check entry conditions
                if buy_signal:
                    entry_price = current_price
                    trade_open = True
                    # Store either timestamp or row index for duration calculation
                    timestamp_val = row.get('timestamp', i)
                    trade_start_timestamp = i if isinstance(timestamp_val, str) else timestamp_val
                    max_price_seen = entry_price
                    min_price_seen = entry_price
        except Exception as e:
            print(f"Error in row {i}: {e}")
            # Skip this row if there's an error
            continue
    
    # Calculate summary statistics
    total_trades = len(trades)
    if total_trades == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'avg_profit': 0,
            'avg_profit_percentage': 0,
            'buy_combo': buy_combo,
            'sell_combo': sell_combo
        }
    
    winning_trades = len([t for t in trades if t['profit'] > 0])
    win_rate = winning_trades / total_trades
    total_profit = sum(t['profit'] for t in trades)
    avg_profit = total_profit / total_trades
    avg_entry_price = sum(t['entry_price'] for t in trades) / total_trades
    avg_profit_percentage = (avg_profit / avg_entry_price * 100) if avg_entry_price > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_profit': avg_profit,
        'avg_profit_percentage': avg_profit_percentage,
        'buy_combo': buy_combo,
        'sell_combo': sell_combo,
        'trades': trades
    }

def test_single_combination(args) -> Dict:
    """
    Test a single combination - designed for parallel processing.
    
    Args:
        args: Tuple of (data, buy_combo, sell_combo, combo_index)
        
    Returns:
        Dict: Backtest results for this combination
    """
    data, buy_combo, sell_combo, combo_index = args
    
    try:
        result = custom_backtest_with_combos(data, buy_combo, sell_combo)
        
        # Add combination info for tracking
        result['combo_index'] = combo_index
        result['buy_combo_str'] = str(buy_combo)
        result['sell_combo_str'] = str(sell_combo)
        
        return result
    except Exception as e:
        print(f"Error testing combo {combo_index}: {e}")
        return None

def test_signal_combinations(data: pd.DataFrame, buy_combinations: List[Dict], sell_combinations: List[Dict], 
                           max_combos: int = None, use_parallel: bool = True, max_workers: int = None, 
                           progress_interval: int = 100, test_all_combinations: bool = False) -> pd.DataFrame:
    """
    Test multiple signal combinations and return results.
    
    Args:
        data (pd.DataFrame): Price data
        buy_combinations (List[Dict]): List of buy signal combinations
        sell_combinations (List[Dict]): List of sell signal combinations
        max_combos (int): Maximum number of combinations to test (None for all)
        use_parallel (bool): Whether to use parallel processing
        max_workers (int): Maximum number of worker processes (defaults to CPU count)
        progress_interval (int): Show progress every N combinations
        test_all_combinations (bool): Test every buy combo with every sell combo
        
    Returns:
        pd.DataFrame: Results DataFrame
    """
    
    # Create combination pairs
    if test_all_combinations:
        # Cartesian product: every buy combo with every sell combo
        print("Creating cartesian product of buy/sell combinations...")
        combination_pairs = list(itertools.product(buy_combinations, sell_combinations))
        
        if max_combos and max_combos < len(combination_pairs):
            # Randomly sample from all possible combinations
            combination_pairs = random.sample(combination_pairs, max_combos)
            print(f"Randomly selected {max_combos} combinations from {len(buy_combinations) * len(sell_combinations):,} total")
    else:
        # Paired combinations: buy combo i with sell combo i
        if max_combos:
            start = random.randint(0, len(buy_combinations) - max_combos)
            end = start + max_combos
            buy_combinations = buy_combinations[start:end]
            sell_combinations = sell_combinations[start:end]
        
        combination_pairs = list(zip(buy_combinations, sell_combinations))
    
    print(f"Testing {len(combination_pairs)} signal combinations...")
    start_time = time.time()
    
    if use_parallel and len(combination_pairs) > 1:
        # Parallel processing
        if max_workers is None:
            max_workers = min(cpu_count(), len(combination_pairs))
        
        print(f"Using parallel processing with {max_workers} workers...")
        
        # Prepare arguments for parallel processing
        args_list = []
        for i, (buy_combo, sell_combo) in enumerate(combination_pairs):
            args_list.append((data, buy_combo, sell_combo, i))
        
        # Use multiprocessing to test combinations in parallel with progress tracking
        with Pool(processes=max_workers) as pool:
            # Use imap for better progress tracking
            results = []
            for i, result in enumerate(pool.imap(test_single_combination, args_list)):
                results.append(result)
                # Show progress based on configured interval or 10%, whichever is more frequent
                actual_interval = max(1, min(progress_interval, len(args_list) // 10))
                if (i + 1) % actual_interval == 0 or i == 0:
                    percentage = ((i + 1) / len(args_list)) * 100
                    print(f"Progress: {i + 1}/{len(args_list)} ({percentage:.1f}%)")
        
        # Filter out None results (failed combinations)
        results = [r for r in results if r is not None]
        
    else:
        # Sequential processing (original method)
        print("Using sequential processing...")
        results = []
        
        for i, (buy_combo, sell_combo) in enumerate(combination_pairs):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"Progress: {i + 1}/{len(combination_pairs)}")
            
            try:
                result = custom_backtest_with_combos(data, buy_combo, sell_combo)
                
                # Add combination info for tracking
                result['combo_index'] = i
                result['buy_combo_str'] = str(buy_combo)
                result['sell_combo_str'] = str(sell_combo)
                
                results.append(result)
            except Exception as e:
                print(f"Error testing combo {i}: {e}")
                continue
    
    # Convert to DataFrame and sort by total profit
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('total_profit', ascending=False)
    
    # Print timing information
    end_time = time.time()
    elapsed_time = end_time - start_time
    combinations_per_second = len(combination_pairs) / elapsed_time if elapsed_time > 0 else 0
    print(f"Completed in {elapsed_time:.2f} seconds ({combinations_per_second:.1f} combinations/sec)")
    
    return results_df

def extract_condition_from_combo_str(combo_str: str, indicator: str) -> str:
    """
    Extract the condition for a specific indicator from a combo string.
    
    Args:
        combo_str (str): String representation of the combo dictionary
        indicator (str): Indicator name to extract condition for
        
    Returns:
        str: Condition string or empty string if not found/ignored
    """
    try:
        # Look for the indicator and its condition
        if f"'{indicator}':" in combo_str:
            # Extract the condition after the indicator
            start_idx = combo_str.find(f"'{indicator}':")
            start_idx = combo_str.find("'", start_idx + len(f"'{indicator}':"))
            if start_idx != -1:
                start_idx += 1
                end_idx = combo_str.find("'", start_idx)
                if end_idx != -1:
                    return combo_str[start_idx:end_idx]
        # If indicator is not found in combo_str, it's ignored
        return ""
    except:
        print(f"Error extracting condition from combo string: {combo_str} for indicator: {indicator}")  
        return ""

def save_combo_results(results_df: pd.DataFrame, filename: str):
    """
    Save combination results to file with proper column structure.
    
    Args:
        results_df (pd.DataFrame): Results DataFrame
        filename (str): Output filename
    """
    if results_df.empty:
        print("No results to save!")
        return
    
    # Create a clean copy for saving
    results_clean = results_df.copy()
    
    
    # Reorder columns for better readability
    column_order = ['file', 'total_trades', 'win_rate', 'total_profit', 
                   'avg_profit', 'avg_profit_percentage', 'combo_index']
    
    # Add buy/sell condition columns in order
    for indicator in ['roc', 'stoch_rsi_k', 'macd_histogram', 'macd_signal', 'macd_line', 'ema']:
        column_order.extend([f'buy_{indicator}', f'sell_{indicator}'])
    
    # Add any remaining columns
    remaining_cols = [col for col in results_clean.columns if col not in column_order]
    column_order.extend(remaining_cols)
    
    # Filter to only existing columns
    column_order = [col for col in column_order if col in results_clean.columns]
    results_clean = results_clean[column_order]
    
    # Save to CSV
    results_clean.to_csv(filename, index=False)
    print(f"Saved {len(results_clean)} results to {filename}")
    print(f"Columns: {list(results_clean.columns)}")

if __name__ == "__main__":
    # Configuration
    ENABLE_PARALLEL = True      # Set to False to disable parallel processing
    MAX_WORKERS = None          # None = auto-detect CPU cores, or set specific number
    MAX_COMBOS = None           # None = test all combinations, or set limit for testing
                               # With filtered combinations: 781,250 total (manageable!)
    MAX_FILES = None            # None = process all files, or set limit for testing
    PROGRESS_INTERVAL = 1000    # Show progress every N combinations (or 10% if smaller)
    TEST_ALL_COMBINATIONS = True  # True = test every buy combo with every sell combo
                                 # False = test each buy combo with corresponding sell combo
    
    # Print system information
    print(f"System detected {cpu_count()} CPU cores")
    if ENABLE_PARALLEL:
        workers = MAX_WORKERS if MAX_WORKERS else cpu_count()
        print(f"Parallel processing enabled with up to {workers} workers")
    else:
        print("Parallel processing disabled - using sequential processing")
    
    # Load data from options folder
    options_folder = 'data/options'
    if not os.path.exists(options_folder):
        print(f"Options folder {options_folder} not found!")
        exit(1)
    
    # Get all CSV files in options folder
    csv_files = [f for f in os.listdir(options_folder) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {options_folder}")
        exit(1)
    
    print(f"Found {len(csv_files)} CSV files in options folder")
    
    # Load signal combinations
    buy_combinations_file = 'signal_combinations_buy_filtered.json'
    sell_combinations_file = 'signal_combinations_sell_filtered.json'
    
    # Check if filtered files exist, otherwise fall back to original
    if os.path.exists(buy_combinations_file) and os.path.exists(sell_combinations_file):
        print(f"Loading filtered signal combinations...")
        try:
            buy_combinations_loaded = load_combinations_from_file(buy_combinations_file)
            sell_combinations_loaded = load_combinations_from_file(sell_combinations_file)
            print(f"Loaded {len(buy_combinations_loaded)} buy combinations (roc > 0 AND stoch_rsi_k > 0)")
            print(f"Loaded {len(sell_combinations_loaded)} sell combinations (roc < 0 AND stoch_rsi_k <= 0)")
            use_filtered_combos = True
        except Exception as e:
            print(f"Error loading filtered combinations: {e}")
            exit(1)
    else:
        print(f"Filtered combination files not found. Run 'python generate_signal_combos.py' to create them.")
        print(f"Falling back to original combinations file...")
        try:
            combinations = load_combinations_from_file('signal_combinations_all_with_ignored.json')
            print(f"Loaded {len(combinations)} combinations from file")
            use_filtered_combos = True
        except FileNotFoundError:
            print("Combination file not found! Run 'python generate_signal_combos.py' first.")
            exit(1)
        except Exception as e:
            print(f"Error loading combinations: {e}")
            exit(1)
    
    # Create buy/sell combinations
    if use_filtered_combos:
        # Use the filtered combinations
        buy_combinations = buy_combinations_loaded
        sell_combinations = sell_combinations_loaded
        print("Using filtered buy/sell combinations")
    else:
        # Use original combinations for both buy and sell
        if TEST_ALL_COMBINATIONS:
            buy_combinations = list(combinations)  # All combinations as buy signals
            sell_combinations = list(combinations)  # All combinations as sell signals
        else:
            buy_combinations, sell_combinations = combinations, combinations
    
    # Calculate total combinations for display
    if TEST_ALL_COMBINATIONS:
        total_combinations = len(buy_combinations) * len(sell_combinations)
        print(f"Buy combinations: {len(buy_combinations)}")
        print(f"Sell combinations: {len(sell_combinations)}")
        print(f"Total combinations to test: {total_combinations:,}")
        
        # Warning for large numbers
        if total_combinations > 1000000:  # 1 million
            estimated_hours = total_combinations / (60 * 60 * 60)  # Very rough estimate
            print(f"⚠️  WARNING: This will test {total_combinations:,} combinations!")
            print(f"⚠️  Estimated time: {estimated_hours:.1f}+ hours (very rough estimate)")
            print("⚠️  Consider setting MAX_COMBOS to a smaller number for testing")
    else:
        print(f"Created {len(buy_combinations)} buy/sell combination pairs")
    
    # Define indicator periods for calculation
    indicator_periods = {
        'ema': 7,
        'vwma': 17,
        'roc': 11,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'stoch_rsi_period': 14,
        'stoch_rsi_k': 3,
        'stoch_rsi_d': 3
    }
    
    all_results = []
    
    # Limit files if specified
    if MAX_FILES:
        csv_files = csv_files[:MAX_FILES]
        print(f"Limited to first {MAX_FILES} files for testing")
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        
        try:
            # Load data
            file_path = os.path.join(options_folder, csv_file)
            data = pd.read_csv(file_path)
            print(f"  Loaded {len(data)} rows")
            
            calculator = IndicatorCalculator()
            data = calculator.calculate_all_indicators(data, indicator_periods, is_option=True)
            
            # Test combinations with configurable parallel processing
            results = test_signal_combinations(
                data, 
                buy_combinations, 
                sell_combinations,
                max_combos=MAX_COMBOS,                    # From configuration
                use_parallel=ENABLE_PARALLEL,             # From configuration
                max_workers=MAX_WORKERS,                  # From configuration
                progress_interval=PROGRESS_INTERVAL,      # From configuration
                test_all_combinations=TEST_ALL_COMBINATIONS  # From configuration
            )
            
            if results.empty:
                print(f"  No results generated for {csv_file}")
                continue
            
            # Add file information to results
            results['file'] = csv_file
            
            # Add buy/sell condition columns
            for indicator in ['roc', 'stoch_rsi_k', 'macd_histogram', 'macd_signal', 'macd_line', 'ema']:
                # Buy condition column
                results[f'buy_{indicator}'] = results['buy_combo_str'].apply(
                    lambda x: extract_condition_from_combo_str(x, indicator)
                )
                
                # Sell condition column
                results[f'sell_{indicator}'] = results['sell_combo_str'].apply(
                    lambda x: extract_condition_from_combo_str(x, indicator)
                )

            
            all_results.append(results)
            print(f"  Generated {len(results)} results")
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # drop buy_combo_str and sell_combo_str
        combined_results = combined_results.drop(columns=['buy_combo_str','buy_combo', 'sell_combo_str', 'sell_combo'])
        # Sort by total profit
        combined_results = combined_results.sort_values('total_profit', ascending=False)
        
        # Save results
        save_combo_results(combined_results, 'options_signal_combo_results.csv')
        
        print(f"\nTotal results: {len(combined_results)}")
        print(f"Results saved to options_signal_combo_results.csv")
    else:
        print("No results generated!")