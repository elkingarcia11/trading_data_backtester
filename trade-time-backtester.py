# Test different trade range combinations
# EX: Only open trades after 10:00am, 10:15am, 10:30am, 11am, 11:30am, 12pm, 12:30pm, 1pm, 1:30pm, 2pm, 2:30pm, 3pm, 3:30pm, 4pm
# EX: Close all trades by 2:30pm, 2:45pm, 3pm, 3:15pm, 3:30pm, 3:45pm, 4pm

import pandas as pd
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
from backtester import backtest

def run_qqq_backtest(args):
    """
    Helper function to run a single QQQ backtest for parallel processing
    """
    qqq_data, qqq_periods, start_time, end_time = args
    
    try:
        qqq_result = backtest(qqq_data, qqq_periods, start_time=start_time, end_time=end_time)
        qqq_result['symbol'] = 'QQQ'
        qqq_result['start_time'] = f"{start_time[0]:02d}:{start_time[1]:02d}"
        qqq_result['end_time'] = f"{end_time[0]:02d}:{end_time[1]:02d}"
        return qqq_result
    except Exception as e:
        print(f"Error testing QQQ {start_time} to {end_time}: {e}")
        return None

def run_spy_backtest(args):
    """
    Helper function to run a single SPY backtest for parallel processing
    """
    spy_data, spy_periods, start_time, end_time = args
    
    try:
        spy_result = backtest(spy_data, spy_periods, start_time=start_time, end_time=end_time)
        spy_result['symbol'] = 'SPY'
        spy_result['start_time'] = f"{start_time[0]:02d}:{start_time[1]:02d}"
        spy_result['end_time'] = f"{end_time[0]:02d}:{end_time[1]:02d}"
        return spy_result
    except Exception as e:
        print(f"Error testing SPY {start_time} to {end_time}: {e}")
        return None

def test_trade_time_combinations():
    """
    Test different trade time combinations and collect results with parallelization
    """
    print("Loading data and calculating indicators...")
    
    qqq_base_indicator_periods = {
        'ema': 6,
        'vwma': 11,
        'roc': 19,
        'roc_of_roc': 16,
        'macd_fast': 22,
        'macd_slow': 36,
        'macd_signal': 30
    }
    spy_base_indicator_periods = {
        'ema': 16,
        'vwma': 13,
        'roc': 3, 
        'roc_of_roc': 3,
        'macd_fast': 9,
        'macd_slow': 12,
        'macd_signal': 12
    }
    
    # Load data with error handling
    try:
        qqq_data = pd.read_csv('data/5m/QQQ.csv')
        spy_data = pd.read_csv('data/5m/SPY.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find data files - {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create a list of all possible start times from 9:30am to 3:50pm
    trade_times = []
    for hour in range(9, 17):
        for minute in range(0, 60, 15):
            if hour == 9 and minute < 30:
                continue
            elif hour == 16 and minute > 0:
                continue
            trade_times.append((hour, minute))
    
    # Create all valid time combinations
    time_combinations = []
    for start_time in trade_times:
        for end_time in trade_times:
            if end_time > start_time:
                time_combinations.append((start_time, end_time))
    
    print(f"Testing {len(trade_times)} start times and {len(trade_times)} end times")
    print(f"Total combinations: {len(time_combinations)}")
    
    # Prepare arguments for parallel processing
    qqq_args_list = []
    spy_args_list = []
    for start_time, end_time in time_combinations:
        qqq_args_list.append((qqq_data, qqq_base_indicator_periods, start_time, end_time))
        spy_args_list.append((spy_data, spy_base_indicator_periods, start_time, end_time))
    
    # Determine number of processes to use
    num_processes = min(cpu_count(), len(time_combinations))
    print(f"Using {num_processes} processes for parallel processing")
    
    # Run parallel backtests
    start_time_total = time.time()
    qqq_results = []
    spy_results = []
    
    # Run QQQ and SPY backtests in parallel
    with Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        for i, qqq_result in enumerate(pool.imap(run_qqq_backtest, qqq_args_list)):
            if qqq_result is not None:
                qqq_results.append(qqq_result)
            
            # Progress tracking every 50 combinations
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time_total
                avg_time_per_test = elapsed / (i + 1)
                remaining_tests = len(qqq_args_list) - (i + 1)
                estimated_remaining = remaining_tests * avg_time_per_test
                print(f"QQQ Progress: {i + 1}/{len(qqq_args_list)} ({(i + 1)/len(qqq_args_list)*100:.1f}%) - ETA: {estimated_remaining/60:.1f} minutes")
    
    # Run SPY backtests
    with Pool(processes=num_processes) as pool:
        for i, spy_result in enumerate(pool.imap(run_spy_backtest, spy_args_list)):
            if spy_result is not None:
                spy_results.append(spy_result)
            
            # Progress tracking every 50 combinations
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time_total
                avg_time_per_test = elapsed / (i + 1)
                remaining_tests = len(spy_args_list) - (i + 1)
                estimated_remaining = remaining_tests * avg_time_per_test
                print(f"SPY Progress: {i + 1}/{len(spy_args_list)} ({(i + 1)/len(spy_args_list)*100:.1f}%) - ETA: {estimated_remaining/60:.1f} minutes")
    
    # Convert results to DataFrame and save
    if qqq_results or spy_results:
        # Convert to DataFrames and sort by total profit
        if qqq_results:
            qqq_df = pd.DataFrame(qqq_results)
            qqq_df = qqq_df.sort_values('total_trade_profit', ascending=False)
        else:
            qqq_df = pd.DataFrame()
            
        if spy_results:
            spy_df = pd.DataFrame(spy_results)
            spy_df = spy_df.sort_values('total_trade_profit', ascending=False)
        else:
            spy_df = pd.DataFrame()
        
        # Save results with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if not qqq_df.empty:
            qqq_df.to_csv(f'data/results/QQQ_trade_time_results_{timestamp}.csv', index=False)
        if not spy_df.empty:
            spy_df.to_csv(f'data/results/SPY_trade_time_results_{timestamp}.csv', index=False)
        
        # Print top 10 results for each symbol
        if not qqq_df.empty:
            print("\nTop 10 Best Performing QQQ Time Combinations:")
            print(qqq_df.head(10)[['start_time', 'end_time', 'total_trade_profit', 'win_rate', 'total_trades']])
        
        if not spy_df.empty:
            print("\nTop 10 Best Performing SPY Time Combinations:")
            print(spy_df.head(10)[['start_time', 'end_time', 'total_trade_profit', 'win_rate', 'total_trades']])
        
        print(f"\nResults saved to:")
        if not qqq_df.empty:
            print(f"  QQQ: data/results/QQQ_trade_time_results_{timestamp}.csv")
        if not spy_df.empty:
            print(f"  SPY: data/results/SPY_trade_time_results_{timestamp}.csv")
        print(f"Total execution time: {(time.time() - start_time_total)/60:.1f} minutes")
    else:
        print("No results generated")

def test_trade_time_combinations_sequential():
    """
    Sequential version for comparison or debugging
    """
    print("Loading data and calculating indicators...")
    
    qqq_base_indicator_periods = {
        'ema': 6,
        'vwma': 11,
        'roc': 19,
        'roc_of_roc': 16,
        'macd_fast': 22,
        'macd_slow': 36,
        'macd_signal': 30
    }
    spy_base_indicator_periods = {
        'ema': 16,
        'vwma': 13,
        'roc': 3, 
        'roc_of_roc': 3,
        'macd_fast': 9,
        'macd_slow': 12,
        'macd_signal': 12
    }
    
    # Load data with error handling
    try:
        qqq_data = pd.read_csv('data/5m/QQQ.csv')
        spy_data = pd.read_csv('data/5m/SPY.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find data files - {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create a list of all possible start times from 9:30am to 3:50pm
    trade_times = []
    for hour in range(9, 16):
        for minute in range(0, 60, 15):
            if hour == 9 and minute < 30:
                continue
            elif hour == 16 and minute > 0:
                continue
            trade_times.append((hour, minute))
    
    print(f"Testing {len(trade_times)} start times and {len(trade_times)} end times")
    print(f"Total combinations: {len(trade_times) * len(trade_times)}")
    
    qqq_results = []
    spy_results = []
    total_combinations = len(trade_times) * len(trade_times)
    current_combination = 0
    start_time_total = time.time()
    
    for start_time in trade_times:
        for end_time in trade_times:
            if end_time > start_time:
                current_combination += 1
                
                # Progress tracking
                if current_combination % 50 == 0:
                    elapsed = time.time() - start_time_total
                    avg_time_per_test = elapsed / current_combination
                    remaining_tests = total_combinations - current_combination
                    estimated_remaining = remaining_tests * avg_time_per_test
                    print(f"Progress: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%) - ETA: {estimated_remaining/60:.1f} minutes")
                
                try:
                    # Test QQQ
                    qqq_result = backtest(qqq_data, qqq_base_indicator_periods, start_time=start_time, end_time=end_time)
                    qqq_result['symbol'] = 'QQQ'
                    qqq_result['start_time'] = f"{start_time[0]:02d}:{start_time[1]:02d}"
                    qqq_result['end_time'] = f"{end_time[0]:02d}:{end_time[1]:02d}"
                    qqq_results.append(qqq_result)
                    
                    # Test SPY
                    spy_result = backtest(spy_data, spy_base_indicator_periods, start_time=start_time, end_time=end_time)
                    spy_result['symbol'] = 'SPY'
                    spy_result['start_time'] = f"{start_time[0]:02d}:{start_time[1]:02d}"
                    spy_result['end_time'] = f"{end_time[0]:02d}:{end_time[1]:02d}"
                    spy_results.append(spy_result)
                    
                except Exception as e:
                    print(f"Error testing {start_time} to {end_time}: {e}")
                    continue
    
    # Convert results to DataFrame and save
    if qqq_results or spy_results:
        # Convert to DataFrames and sort by total profit
        if qqq_results:
            qqq_df = pd.DataFrame(qqq_results)
            qqq_df = qqq_df.sort_values('total_trade_profit', ascending=False)
        else:
            qqq_df = pd.DataFrame()
            
        if spy_results:
            spy_df = pd.DataFrame(spy_results)
            spy_df = spy_df.sort_values('total_trade_profit', ascending=False)
        else:
            spy_df = pd.DataFrame()
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if not qqq_df.empty:
            qqq_df.to_csv(f'data/results/QQQ_trade_time_results_sequential_{timestamp}.csv', index=False)
        if not spy_df.empty:
            spy_df.to_csv(f'data/results/SPY_trade_time_results_sequential_{timestamp}.csv', index=False)
        
        # Print top 10 results for each symbol
        if not qqq_df.empty:
            print("\nTop 10 Best Performing QQQ Time Combinations:")
            print(qqq_df.head(10)[['start_time', 'end_time', 'total_trade_profit', 'win_rate', 'total_trades']])
        
        if not spy_df.empty:
            print("\nTop 10 Best Performing SPY Time Combinations:")
            print(spy_df.head(10)[['start_time', 'end_time', 'total_trade_profit', 'win_rate', 'total_trades']])
        
        print(f"\nResults saved to:")
        if not qqq_df.empty:
            print(f"  QQQ: data/results/QQQ_trade_time_results_sequential_{timestamp}.csv")
        if not spy_df.empty:
            print(f"  SPY: data/results/SPY_trade_time_results_sequential_{timestamp}.csv")
        print(f"Total execution time: {(time.time() - start_time_total)/60:.1f} minutes")
    else:
        print("No results generated")

if __name__ == "__main__":
    # Use parallel processing by default
    test_trade_time_combinations()
    
    # Uncomment the line below to use sequential processing instead
    # test_trade_time_combinations_sequential()