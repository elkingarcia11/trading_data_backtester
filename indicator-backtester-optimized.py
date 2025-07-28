#!/usr/bin/env python3
"""
Optimized version of indicator backtester with significant performance improvements
"""

import pandas as pd
import numpy as np
from backtester import backtest, run_single_backtest
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import os
import shutil
from itertools import product
import gc
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Configure logging for better performance monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache for file data to avoid repeated I/O
_data_cache = {}

class ProgressTracker:
    """Class to track progress and calculate ETA"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.completed_items = 0
        self.start_time = time.time()
        self.description = description
        self.last_update_time = self.start_time
        self.update_interval = 5  # Update every 5 seconds
    
    def update(self, completed: int = 1):
        """Update progress and log ETA if enough time has passed"""
        self.completed_items += completed
        current_time = time.time()
        
        # Only update if enough time has passed to avoid spam
        if current_time - self.last_update_time >= self.update_interval:
            self._log_progress(current_time)
            self.last_update_time = current_time
    
    def _log_progress(self, current_time: float):
        """Log current progress with ETA"""
        elapsed_time = current_time - self.start_time
        
        if self.completed_items > 0:
            # Calculate progress percentage
            progress_pct = (self.completed_items / self.total_items) * 100
            
            # Calculate ETA
            avg_time_per_item = elapsed_time / self.completed_items
            remaining_items = self.total_items - self.completed_items
            eta_seconds = avg_time_per_item * remaining_items
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            
            # Format elapsed and ETA times
            elapsed_str = self._format_time(elapsed_time)
            eta_str = eta_time.strftime("%H:%M:%S")
            
            # Create a simple progress bar
            bar_length = 30
            filled_length = int(bar_length * self.completed_items // self.total_items)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            logger.info(f"{self.description}: [{bar}] {self.completed_items}/{self.total_items} "
                       f"({progress_pct:.1f}%) - Elapsed: {elapsed_str} - ETA: {eta_str}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def finish(self):
        """Log final completion time"""
        total_time = time.time() - self.start_time
        total_time_str = self._format_time(total_time)
        logger.info(f"{self.description}: COMPLETED in {total_time_str}")

def fetch_options_data_optimized(file_path: str, symbol: str) -> List[pd.DataFrame]:
    """Optimized options data fetching with caching"""
    cache_key = f"{file_path}_{symbol}"
    
    if cache_key in _data_cache:
        return _data_cache[cache_key]
    
    files = [f for f in os.listdir(file_path) if f.startswith(symbol) and f.endswith('.csv')]
    
    # Use list comprehension and parallel reading for better performance
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(os.path.join(file_path, file), low_memory=False)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {file}: {e}")
    
    _data_cache[cache_key] = dfs
    return dfs

def generate_combinations_optimized(base_periods: Dict) -> List[Dict]:
    """
    Optimized combination generator with smart filtering and reduced search space
    """
    # Reduced ranges for faster testing - adjust based on your needs
    ranges = {
        'ema': range(7, 14),     
        'vwma': range(12, 21),   
        'roc': range(3, 12, 2),      
        'roc_of_roc': range(7, 18, 2), 
        'stoch_rsi_period': range(3, 21, 2),
        'stoch_rsi_k': range(3, 21, 2),
        'stoch_rsi_d': range(3, 21, 2),
    }
    
    # Generate combinations using itertools.product for better performance
    combinations = []
    
    # Smart filtering: only generate combinations that make sense
    for ema, vwma, roc, roc_of_roc, stoch_rsi_period, stoch_rsi_k, stoch_rsi_d in product(
        ranges['ema'], ranges['vwma'], ranges['roc'], ranges['roc_of_roc'],
        ranges['stoch_rsi_period'], ranges['stoch_rsi_k'], ranges['stoch_rsi_d']
    ):
        if ema == vwma:
            continue
            
        combo = base_periods.copy()
        combo.update({
            'ema': ema,
            'vwma': vwma, 
            'roc': roc,
            'roc_of_roc': roc_of_roc,
            'stoch_rsi_period': stoch_rsi_period,
            'stoch_rsi_k': stoch_rsi_k,
            'stoch_rsi_d': stoch_rsi_d
        })
        combinations.append(combo)
    
    logger.info(f"Generated {len(combinations)} combinations (optimized from ~2.7M)")
    return combinations

def preprocess_data_optimized(data: pd.DataFrame, is_options: bool = False) -> pd.DataFrame:
    """Optimized data preprocessing with vectorized operations"""
    # Convert datetime/timestamp once and cache
    if is_options:
        # Options data uses 'timestamp' column
        if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S')
            except:
                data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        
        # Create datetime column for consistency
        if 'datetime' not in data.columns:
            data['datetime'] = data['timestamp']
    else:
        # Regular stock data uses 'datetime' column
        if 'datetime' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['datetime']):
            try:
                data['datetime'] = pd.to_datetime(data['datetime'], format='mixed')
            except:
                data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    
    # Extract date using vectorized operations
    if 'date' not in data.columns:
        try:
            data['date'] = data['datetime'].dt.date.astype(str)
        except:
            # Fallback: extract date from datetime string
            data['date'] = data['datetime'].astype(str).str.split(' ').str[0]
    
    # Ensure numeric columns are properly typed
    if is_options:
        # Options data uses last_price and volume
        numeric_cols = ['last_price', 'volume']
    else:
        # Regular stock data uses OHLCV
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data

def chunk_combinations(combinations: List[Dict], chunk_size: int = 1000) -> List[List[Dict]]:
    """Split combinations into chunks for better memory management"""
    return [combinations[i:i + chunk_size] for i in range(0, len(combinations), chunk_size)]

def process_chunk_optimized(args: Tuple) -> List[Dict]:
    """Optimized chunk processing function"""
    data, combinations_chunk, is_options = args
    
    results = []
    for combo in combinations_chunk:
        try:
            # Ensure data is a copy to avoid issues in parallel processing
            data_copy = data.copy()
            result = backtest(data_copy, combo, is_options=is_options)
            results.append(result)
        except Exception as e:
            logger.warning(f"Backtest failed for combo {combo}: {e}")
            continue
    
    return results

def test_indicator_period_combinations_optimized(
    data: pd.DataFrame, 
    timeframe: str, 
    symbol: str, 
    test_combinations: List[Dict], 
    is_options: bool = False, 
    use_parallel: bool = True, 
    max_workers: Optional[int] = None,
    chunk_size: int = 1000
) -> pd.DataFrame:
    """
    Optimized version of test_indicator_period_combinations with better memory management
    """
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    logger.info(f"Processing {len(test_combinations)} combinations with {max_workers} workers")
    
    # Preprocess data once
    data = preprocess_data_optimized(data.copy(), is_options)
    
    # Split combinations into chunks for better memory management
    chunks = chunk_combinations(test_combinations, chunk_size)
    logger.info(f"Split into {len(chunks)} chunks of size {chunk_size}")
    
    # Initialize progress tracker for chunks
    chunk_tracker = ProgressTracker(len(chunks), f"Chunks for {symbol} {timeframe}")
    
    all_results = []
    
    if use_parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunks instead of individual combinations
            futures = []
            for chunk in chunks:
                future = executor.submit(process_chunk_optimized, (data, chunk, is_options))
                futures.append(future)
            
            # Collect results with progress tracking
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    chunk_tracker.update(1)
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}")
                    chunk_tracker.update(1)
    else:
        # Sequential processing with chunks
        for i, chunk in enumerate(chunks):
            chunk_results = process_chunk_optimized((data, chunk, is_options))
            all_results.extend(chunk_results)
            chunk_tracker.update(1)
    
    chunk_tracker.finish()
    
    # Save results efficiently
    if all_results:
        return save_results_optimized(all_results, timeframe, symbol, is_options)
    
    return pd.DataFrame()

def save_results_optimized(results: List[Dict], timeframe: str, symbol: str, is_options: bool) -> pd.DataFrame:
    """Optimized results saving with better memory management"""
    
    result_df = pd.DataFrame(results)
    
    # Determine file path
    if is_options:
        options_dir = 'data/results/options'
        os.makedirs(options_dir, exist_ok=True)
        filename = f'{symbol}_{results[0].get("contract_type", "unknown")}_{results[0].get("strike_price", "unknown")}.csv'
        filepath = f'{options_dir}/{filename}'
    else:
        timeframe_dir = f'data/results/{timeframe}'
        os.makedirs(timeframe_dir, exist_ok=True)
        filepath = f'{timeframe_dir}/{symbol}.csv'
    
    # Efficient file handling
    try:
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            # Clean up unnamed columns efficiently
            existing_df = existing_df.loc[:, ~existing_df.columns.str.contains('^Unnamed')]
            
            if set(existing_df.columns) == set(result_df.columns):
                result_df = pd.concat([existing_df, result_df], ignore_index=True)
                logger.info(f"Appended {len(results)} new results to existing {len(existing_df)} results")
            else:
                logger.warning("Column mismatch detected. Using new results only.")
    except FileNotFoundError:
        logger.info(f"Created new file with {len(results)} results")
    
    # Sort efficiently
    if not result_df.empty:
        result_df = result_df.sort_values(['average_trade_profit'], ascending=[False])
        result_df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(result_df)} results to {filepath}")
    
    return result_df

def test_intraday_periods_optimized(
    is_options: bool = False, 
    use_parallel: bool = True, 
    max_workers: Optional[int] = None,
    chunk_size: int = 1000
):
    """Optimized version of test_intraday_periods"""
    
    start_time = time.time()
    
    if is_options:
        if os.path.exists('data/results/options/'):
            shutil.rmtree('data/results/options/')
        os.makedirs('data/results/options/', exist_ok=True)

    # Load symbols once
    with open('symbols.txt', 'r') as f:
        symbols = f.read().splitlines()
    
    logger.info(f"Processing symbols: {symbols}")

    # Base configurations
    base_configs = {
        'QQQ': {},
        'SPY': {}
    }

    # Initialize overall progress tracker
    overall_tracker = ProgressTracker(len(symbols), "Overall Symbol Processing")
    
    for symbol in symbols:
        symbol_start = time.time()
        logger.info(f"Starting processing for {symbol}")
        
        base_periods = base_configs.get(symbol, {})
        test_combinations = generate_combinations_optimized(base_periods)
        
        test_intraday_periods_for_optimized(
            symbol, is_options, test_combinations, use_parallel, max_workers, chunk_size
        )
        
        symbol_time = time.time() - symbol_start
        logger.info(f"Completed {symbol} in {symbol_time:.2f} seconds")
        
        # Update overall progress
        overall_tracker.update(1)
        
        # Clear cache to free memory
        gc.collect()
    
    overall_tracker.finish()
    total_time = time.time() - start_time
    logger.info(f"Total processing time: {total_time:.2f} seconds")

def test_intraday_periods_for_optimized(
    symbol: str, 
    is_options: bool, 
    test_combinations: List[Dict], 
    use_parallel: bool, 
    max_workers: Optional[int],
    chunk_size: int
):
    """Optimized version of test_intraday_periods_for"""
    
    if is_options:
        dfs = fetch_options_data_optimized('data/options/', symbol)
        logger.info(f"Processing {len(dfs)} options dataframes for {symbol}")
        
        # Initialize progress tracker for options dataframes
        options_tracker = ProgressTracker(len(dfs), f"Options Dataframes for {symbol}")
        
        if use_parallel:
            # Process options dataframes in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for df_idx, df in enumerate(dfs):
                    logger.info(f"Processing options dataframe {df_idx + 1}/{len(dfs)}")
                    future = executor.submit(
                        test_indicator_period_combinations_optimized,
                        df, "options", symbol, test_combinations, True, False, None, chunk_size
                    )
                    futures.append(future)
                
                # Collect results
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        options_tracker.update(1)
                    except Exception as e:
                        logger.error(f"Error processing options dataframe {i+1}: {e}")
                        options_tracker.update(1)
        else:
            # Sequential processing for options dataframes
            for df_idx, df in enumerate(dfs):
                logger.info(f"Processing options dataframe {df_idx + 1}/{len(dfs)}")
                try:
                    result = test_indicator_period_combinations_optimized(
                        df, "options", symbol, test_combinations, True, False, None, chunk_size
                    )
                    options_tracker.update(1)
                except Exception as e:
                    logger.error(f"Error processing options dataframe {df_idx + 1}: {e}")
                    options_tracker.update(1)
        
        options_tracker.finish()
    else:
        # Load timeframes once
        with open('timeframes.txt', 'r') as f:
            timeframes = f.read().splitlines()
        
        # Initialize progress tracker for timeframes
        timeframe_tracker = ProgressTracker(len(timeframes), f"Timeframes for {symbol}")
        
        for timeframe in timeframes:
            timeframe_start = time.time()
            
            # Clear existing results efficiently
            result_file = f'data/results/{timeframe}/{symbol}.csv'
            if os.path.exists(result_file):
                os.remove(result_file)
                logger.info(f"Cleared existing results: {result_file}")

            logger.info(f"Processing {symbol} - {timeframe}")
            
            # Load data once
            data = pd.read_csv(f'data/{timeframe}/{symbol}.csv', low_memory=False)
            data = preprocess_data_optimized(data, is_options)
            
            # Regular stock data - group by day
            if 'date' not in data.columns:
                try:
                    data['date'] = data['datetime'].dt.date.astype(str)
                except:
                    # Fallback: extract date from datetime string
                    data['date'] = data['datetime'].astype(str).str.split(' ').str[0]
            unique_days = data['date'].unique()
            
            logger.info(f"Testing {len(unique_days)} days for {symbol} {timeframe}")
            
            # Initialize progress tracker for days
            day_tracker = ProgressTracker(len(unique_days), f"Days for {symbol} {timeframe}")
            
            if use_parallel:
                # Process days in parallel with better memory management
                day_data_list = []
                for day in unique_days:
                    day_data = data[data['date'] == day].drop(columns=['date']).copy()
                    day_data_list.append((day_data, day))
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for day_data, day in day_data_list:
                        future = executor.submit(
                            test_indicator_period_combinations_optimized,
                            day_data, timeframe, symbol, test_combinations, False, False, None, chunk_size
                        )
                        futures.append((future, day))
                    
                    # Collect results
                    for future, day in as_completed(futures):
                        try:
                            result = future.result()
                            day_tracker.update(1)
                        except Exception as e:
                            logger.error(f"Error processing {day}: {e}")
                            day_tracker.update(1)
            else:
                # Sequential processing
                for day_idx, day in enumerate(unique_days, 1):
                    logger.info(f"Day {day_idx}/{len(unique_days)}: {day}")
                    day_data = data[data['date'] == day].drop(columns=['date']).copy()
                    test_indicator_period_combinations_optimized(
                        day_data, timeframe, symbol, test_combinations, False, False, None, chunk_size
                    )
                    day_tracker.update(1)
            
            day_tracker.finish()
            timeframe_time = time.time() - timeframe_start
            logger.info(f"Completed {timeframe} in {timeframe_time:.2f} seconds")
            timeframe_tracker.update(1)
        
        timeframe_tracker.finish()

# Example usage
if __name__ == "__main__":
    # Test with optimized settings
    test_intraday_periods_optimized(
        is_options=True, 
        use_parallel=True, 
        max_workers=4,  # Conservative to avoid memory issues
        chunk_size=500  # Smaller chunks for better memory management
    ) 