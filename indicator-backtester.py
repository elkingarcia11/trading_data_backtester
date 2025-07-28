import pandas as pd

from backtester import backtest, run_single_backtest
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import os
import shutil

def fetch_options_data(file_path, symbol: str):
    # Get every csv file in file_path that starts with symbol using pandas
    files = os.listdir(file_path)
    files = [file for file in files if file.startswith(symbol)]
    # Read the files
    dfs = []
    for file in files:
        df = pd.read_csv(os.path.join(file_path, file))
        dfs.append(df)
    return dfs

def test_ema_vwma_roc_and_roc_of_roc_periods(base_indicator_periods):   
    test_combinations = []
    for ema in range(8,14):
        for vwma in range(12,22):
            if ema != vwma:
                for roc in range(3,14):
                    for roc_of_roc in range(7,21):
                        base_indicator_periods['ema'] = ema 
                        base_indicator_periods['vwma'] = vwma
                        base_indicator_periods['roc'] = roc
                        base_indicator_periods['roc_of_roc'] = roc_of_roc
                        test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_roc_and_roc_of_roc_periods(base_indicator_periods):
    test_combinations = []
    for roc in range(3,21):
        for roc_of_roc in range(3,21):
            base_indicator_periods['roc'] = roc
            base_indicator_periods['roc_of_roc'] = roc_of_roc
            test_combinations.append(base_indicator_periods.copy())
    return test_combinations
    
def test_roc_periods(base_indicator_periods):
    test_combinations = []
    for roc in range(3,21):
        base_indicator_periods['roc'] = roc
        test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_ema_vwma_periods(base_indicator_periods):
    test_combinations = []
    for ema in range(5,21):
        for vwma in range(5,21):
            if ema != vwma:
                base_indicator_periods['ema'] = ema
                base_indicator_periods['vwma'] = vwma
                test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_macd_periods(base_indicator_periods):
    test_combinations = [base_indicator_periods.copy()]
    for fast in range(5,45):
        for slow in range(6,45):
            for signal in range(2,45):
                if fast < slow:
                    base_indicator_periods['macd_fast'] = fast  
                    base_indicator_periods['macd_slow'] = slow
                    base_indicator_periods['macd_signal'] = signal
                    test_combinations.append(base_indicator_periods.copy())
    return test_combinations
    
def test_stoch_vs_macd(base_indicator_periods: dict) -> list[dict]:

    test_combinations = []

    # one with macd and with roc of roc
    test_combinations.append(base_indicator_periods.copy())
    
    # one with macd and without roc of roc
    base_indicator_periods.pop('roc_of_roc')
    test_combinations.append(base_indicator_periods.copy())

    # one with stoch field and with roc of roc
    base_indicator_periods['roc_of_roc'] = 10
    base_indicator_periods['stoch_rsi_period'] = 14
    base_indicator_periods['stoch_rsi_k'] = 12
    base_indicator_periods['stoch_rsi_d'] = 12
    test_combinations.append(base_indicator_periods.copy())
    # one with stoch field but without roc of roc
    base_indicator_periods.pop('roc_of_roc')
    test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_stoch_periods(base_indicator_periods: dict) -> list[dict]:
    test_combinations = [base_indicator_periods.copy()]
    for period in range(2,20):
        for k in range(2,20):
            for d in range(2,20):
                base_indicator_periods['stoch_rsi_period'] = period
                base_indicator_periods['stoch_rsi_k'] = k
                base_indicator_periods['stoch_rsi_d'] = d
                test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_indicator_period_combinations(data: pd.DataFrame, timeframe: str, symbol: str, test_combinations: list[dict], is_options: bool = False, use_parallel: bool = True, max_workers: int | None = None) -> pd.DataFrame:
    """
    Test different indicator period combinations with optional parallelization
    
    Args:
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of worker processes (defaults to CPU count)
    """

    if use_parallel:
        # Use parallel processing
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        print(f"Running {len(test_combinations)} backtests using {max_workers} parallel workers...")
        time.sleep(2)
        results = []
        completed = 0
        total = len(test_combinations)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all backtests
            future_to_combo = {executor.submit(run_single_backtest, (data, combo, is_options)): combo for combo in test_combinations}
            
            # Collect results as they complete
            for future in as_completed(future_to_combo):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 10 == 0 or completed == total:  # Show progress every 10 or on completion
                        print(f"Progress: {completed}/{total} backtests completed ({completed/total*100:.1f}%)")
                except Exception as exc:
                    combo = future_to_combo[future]
                    print(f'Backtest generated an exception: {exc}')
                    completed += 1
    else:
        # Sequential processing (original method)
        print(f"Running {len(test_combinations)} backtests sequentially...")
        results = []
        completed = 0
        total = len(test_combinations)
        
        for indicator_periods in test_combinations:
            result = backtest(data, indicator_periods, is_options=is_options)
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == total:  # Show progress every 10 or on completion
                print(f"Progress: {completed}/{total} backtests completed ({completed/total*100:.1f}%)")

    # Save all results to CSV
    if results:
        result_df = pd.DataFrame(results)
        
        # Determine the file path based on whether it's options data or not
        if is_options:
            options_dir = 'data/results/options'
            os.makedirs(options_dir, exist_ok=True)
            filename = f'{symbol}_{data["contract_type"].iloc[0]}_{data["strike_price"].iloc[0]}.csv'
            filepath = f'{options_dir}/{filename}'
        else:
            timeframe_dir = f'data/results/{timeframe}'
            os.makedirs(timeframe_dir, exist_ok=True)
            filepath = f'{timeframe_dir}/{symbol}.csv'
        
        # Always append to existing results and sort at the end
        try:
            existing_df = pd.read_csv(filepath)
            
            # Clean up any unnamed columns from corrupted files
            existing_df = existing_df.loc[:, ~existing_df.columns.str.contains('^Unnamed')]
            
            # Ensure both DataFrames have the same columns
            if set(existing_df.columns) != set(result_df.columns):
                print(f"Warning: Column mismatch detected. Recreating file with new structure.")
                combined_df = result_df
            else:
                combined_df = pd.concat([existing_df, result_df], ignore_index=True)
                print(f"Appended {len(results)} new results to existing {len(existing_df)} results")
        except FileNotFoundError:
            combined_df = result_df
            print(f"Created new file with {len(results)} results")
        
        # Sort by appropriate metric
        sorted_df = combined_df.sort_values(['average_trade_profit'], ascending=[False])
        print(f"Saved {len(sorted_df)} total results to {filepath} (sorted by total profit)")
        
        if is_options:
            # Extract trades with total_trades > 5
            new_trades = sorted_df[sorted_df['total_trades'] > 5].copy()
            
            combined_filepath = f'data/results/options/{symbol}.csv'
            if os.path.exists(combined_filepath):
                existing_combined_df = pd.read_csv(combined_filepath)
                combined_df = pd.concat([existing_combined_df, new_trades], ignore_index=True)
                sorted_combined_df = combined_df.sort_values(['average_trade_profit'], ascending=[False]) 
            else:
                sorted_combined_df = new_trades
            sorted_combined_df.to_csv(combined_filepath, mode='w', index=False)
        
        # Save sorted combined results
        sorted_df.to_csv(filepath, mode='w', index=False)
        print(f"Saved {len(sorted_df)} total results to {filepath} (sorted by total profit)")
        
        return sorted_df
    return pd.DataFrame() if results else pd.DataFrame()

def test_with_and_without_roc_of_roc(base_indicator_periods):
    test_combinations = [] 
    test_combinations.append(base_indicator_periods.copy())
    base_indicator_periods.pop('roc_of_roc')
    test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_different_roc_periods(base_indicator_periods):
    test_combinations = []
    for roc in range(14,21):
        base_indicator_periods['roc'] = roc
        test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_ema_vwma_roc_roc_of_roc_macd_periods(base_indicator_periods):
    test_combinations = []
    for ema in range(8,16,2):
        for vwma in range(12,24,2):
            for roc in range(3,16,2):
                for roc_of_roc in range(7,24,2):
                    for macd_fast in range(5,46,2):
                        for macd_slow in range(6,45,2):
                            for macd_signal in range(2,45):
                                base_indicator_periods['ema'] = ema
                                base_indicator_periods['vwma'] = vwma
                                base_indicator_periods['roc'] = roc
                                base_indicator_periods['roc_of_roc'] = roc_of_roc
                                base_indicator_periods['macd_fast'] = macd_fast
                                base_indicator_periods['macd_slow'] = macd_slow
                                base_indicator_periods['macd_signal'] = macd_signal
                                test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_intraday_periods(is_options: bool = False, use_parallel: bool = True, max_workers: int = None):  
    if is_options:
        # Remove directory data/results/options/
        if os.path.exists('data/results/options/'):
            shutil.rmtree('data/results/options/')
        os.makedirs('data/results/options/', exist_ok=True)

    qqq_base_intraday_indicator_periods_1m = {
    }

    spy_base_intraday_indicator_periods_1m = {
    }

    symbols = open('symbols.txt', 'r').read().splitlines()
    print("SYMBOLS: ", symbols)

    is_random_sampling = False
    is_testing_finalized_indicator_periods = False

    # Testing different indicator period combinations
    for symbol in symbols:
        if symbol == 'QQQ':
            base_indicator_periods = qqq_base_intraday_indicator_periods_1m 
            if is_testing_finalized_indicator_periods:
                test_combinations = [qqq_base_intraday_indicator_periods_1m]
            else:
                test_combinations = test_ema_vwma_roc_roc_of_roc_macd_periods(base_indicator_periods)
        else: 
            base_indicator_periods = spy_base_intraday_indicator_periods_1m
            if is_testing_finalized_indicator_periods:
                test_combinations = [spy_base_intraday_indicator_periods_1m]
            else:
                test_combinations = test_ema_vwma_roc_roc_of_roc_macd_periods(base_indicator_periods)
        test_intraday_periods_for(symbol, is_options, test_combinations, is_random_sampling, use_parallel, max_workers)

def test_intraday_periods_for(symbol, is_options, test_combinations, is_random_sampling, use_parallel, max_workers):
    if not is_options:
        timeframes = open('timeframes.txt', 'r').read().splitlines()
    if is_options:
        dfs = fetch_options_data(f'data/options/', symbol)
        total_dfs = len(dfs)
        print(f"Processing {total_dfs} options dataframes for {symbol}")
    else:
        dfs = pd.read_csv(f'data/{timeframe}/{symbol}.csv')
    # If is_options, fetch the data from the options directory
    if is_options:
        dfs = fetch_options_data(f'data/options/', symbol)
        total_dfs = len(dfs)
        print(f"Processing {total_dfs} options dataframes for {symbol}")
        
        # Process each dataframe sequentially, but parallelize combinations within each dataframe
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for df_idx, df in enumerate(dfs):
                print(f"  Processing options dataframe {df_idx + 1}/{total_dfs}") # Process days in parallel
                future = executor.submit(
                    test_indicator_period_combinations, 
                    df, 
                    "options", 
                    symbol, 
                    test_combinations, 
                    True,
                    False,  # Don't use parallel within parallel
                    None
                )
                futures.append(future)
            
            # Collect results with progress tracking
            completed = 0
            for future in futures:
                try:
                    result = future.result()
                    completed += 1
                    print(f"  Completed {completed}/{total_dfs}")
                except Exception as e:
                    print(f"  Error processing {e}")
    else:
        for timeframe in timeframes:
            # Erase data/results/timeframe/symbol.csv if it exists
            result_file = f'data/results/{timeframe}/{symbol}.csv'
            if os.path.exists(result_file):
                os.remove(result_file)
                print(f"Cleared existing results file: {result_file}")

            print(f"\nProcessing {symbol} - {timeframe}")
            
            # Load data once
            data = pd.read_csv(f'data/{timeframe}/{symbol}.csv')
            
            if is_random_sampling:
                # Sample a random day for faster testing
                data = sample_random_data(data, symbol=symbol, timeframe=timeframe)
                test_indicator_period_combinations(data, timeframe, symbol, test_combinations, True, use_parallel, max_workers)
            else:
                # Extract date once and group by day
                data['date'] = data['datetime'].str.split(' ').str[0]
                unique_days = data['date'].unique()
                total_days = len(unique_days)
                
                print(f"Testing {total_days} days for {symbol} {timeframe}")
                
                if use_parallel:
                    # Process days in parallel
                    import multiprocessing
                    
                    if max_workers is None:
                        max_workers = multiprocessing.cpu_count()
                    
                    print(f"Using {max_workers} parallel workers")
                    
                    # Prepare data for parallel processing
                    day_data_list = []
                    for day in unique_days:
                        data_day = data[data['date'] == day].copy()
                        data_day = data_day.drop(columns=['date'])
                        day_data_list.append((data_day, day))
                    
                    # Process days in parallel
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for data_day, day in day_data_list:
                            future = executor.submit(
                                test_indicator_period_combinations, 
                                data_day, 
                                timeframe, 
                                symbol, 
                                test_combinations, 
                                True,
                                False,  # Don't use parallel within parallel
                                None
                            )
                            futures.append((future, day))
                        
                        # Collect results with progress tracking
                        completed = 0
                        for future, day in futures:
                            try:
                                result = future.result()
                                completed += 1
                                print(f"  Completed {completed}/{total_days}: {day}")
                            except Exception as e:
                                print(f"  Error processing {day}: {e}")
                else:
                    # Process each day sequentially
                    for day_idx, day in enumerate(unique_days, 1):
                        print(f"  Day {day_idx}/{total_days}: {day}")
                        data_day = data[data['date'] == day].copy()
                        data_day = data_day.drop(columns=['date'])  # Clean up temporary column
                        
                        # Test indicator combinations for this day
                        test_indicator_period_combinations(data_day, timeframe, symbol, test_combinations, True, False, None)
                
def test_interday_periods():
    qqq_base_interday_indicator_periods = {
    }
    spy_base_interday_indicator_periods = {
    }
    symbols = open('symbols.txt', 'r').read().splitlines()
    timeframes = open('timeframes.txt', 'r').read().splitlines()

    # Testing different indicator period combinations
    for symbol in symbols:
        if symbol == 'QQQ':
            base_indicator_periods = qqq_base_interday_indicator_periods
            test_combinations = test_ema_vwma_roc_and_roc_of_roc_periods(base_indicator_periods)
        else:
            base_indicator_periods = spy_base_interday_indicator_periods
            test_combinations = test_ema_vwma_roc_and_roc_of_roc_periods(base_indicator_periods)
        for timeframe in timeframes:
            data = pd.read_csv(f'data/{timeframe}/{symbol}.csv')
            test_indicator_period_combinations(data, timeframe, symbol, test_combinations, False)

def sample_random_data(data: pd.DataFrame, symbol: str = 'SPY', timeframe: str = '5m'):
    """
    Sample a random day from the dataset (intraday sampling)
    
    Args:
        data: DataFrame with datetime column
        symbol: Symbol name for logging
        timeframe: Timeframe for logging
    """
    import random
    
    # Extract just the date part (YYYY-MM-DD) from datetime strings like '2025-06-03 09:30:00 EDT'
    if 'datetime' in data.columns:
        data['date'] = data['datetime'].str.split(' ').str[0]
 
    else:
        # Fallback to timestamp if datetime column doesn't exist
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
        data['date'] = data['datetime'].dt.date
        print("Using timestamp column for date extraction")
    
    # Always sample a random day (all rows from that day)
    available_dates = data['date'].unique()
    random_date = random.choice(available_dates)
    sampled_data = data[data['date'] == random_date].copy()
    print(f"Sampled data for {symbol} {timeframe} on date: {random_date} ({len(sampled_data)} rows)")

    
    # Drop the temporary date column
    sampled_data = sampled_data.drop(columns=['date'])
    return sampled_data

if __name__ == '__main__':
    # Use parallel processing with all CPU cores
    test_intraday_periods(is_options=True, use_parallel=True, max_workers=10)