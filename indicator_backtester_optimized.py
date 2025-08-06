#!/usr/bin/env python3
"""
Optimized version of indicator backtester with significant performance improvements
"""

import pandas as pd
import numpy as np
from backtester import backtest
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
from indicator_config import INDICATOR_RANGES

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

def analyze_available_indicators(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Analyze which indicators are available in the data after calculation
    Maps parameter names to actual column names that get created
    """
    # Map parameter names to actual column names that get created
    parameter_to_column_mapping = {
        'ema': 'ema',
        'ema_fast': 'ema_fast',  # ema_fast creates ema_fast column
        'ema_slow': 'ema_slow',  # ema_slow creates ema_slow column
        'vwma': 'vwma',
        'vwma_fast': 'vwma_fast',
        'vwma_slow': 'vwma_slow',
        'rsi': 'rsi',
        'stoch_rsi_k': 'stoch_rsi_k',
        'stoch_rsi_d': 'stoch_rsi_d',
        'stoch_rsi_period': 'stoch_rsi_k',  # stoch_rsi_period creates stoch_rsi_k and stoch_rsi_d
        'macd_fast': 'macd_line',  # macd_fast creates macd_line, macd_signal
        'macd_slow': 'macd_line',  # macd_slow creates macd_line, macd_signal
        'macd_signal': 'macd_line',  # macd_signal creates macd_line, macd_signal
        'roc': 'roc',
        'roc_of_roc': 'roc_of_roc',
        'sma': 'sma',
        'volatility': 'volatility',
        'price_change': 'price_change',
        'bollinger_period': 'bollinger_upper',  # bollinger_period creates bollinger_upper, bollinger_lower, bollinger_bands_width
        'bollinger_std': 'bollinger_upper',  # bollinger_std creates bollinger_upper, bollinger_lower, bollinger_bands_width
        'atr': 'atr'
    }
    
    # Use the indicator configuration to define what to look for
    all_indicators = {indicator: False for indicator in INDICATOR_RANGES.keys()}
    
    # Check which indicators are actually present in the data
    for indicator in all_indicators.keys():
        if indicator in parameter_to_column_mapping:
            column_name = parameter_to_column_mapping[indicator]
            if column_name in data.columns:
                all_indicators[indicator] = True
        else:
            # For parameters that don't directly map to columns, assume they're available
            # These are input parameters for calculations, not output columns
            if indicator in ['stoch_rsi_period', 'macd_fast', 'macd_slow', 'macd_signal']:
                all_indicators[indicator] = True
    
    # Log available indicators
    available = [k for k, v in all_indicators.items() if v]
    logger.info(f"Available indicators in data: {available}")
    
    return all_indicators

def generate_combinations_optimized(ranges: Dict) -> List[Dict]:
    """
    Optimized combination generator with smart filtering and reduced search space
    Only generates combinations for indicators that are actually present in the data
    """
    
    # If no available indicators provided, use all indicators from ranges
    if not ranges:
        logger.warning("No indicators available for combination generation")
        return []
    
    # Generate combinations using itertools.product for better performance
    combinations = []
    # Get the indicator names for product generation
    indicator_names = list(ranges.keys())
    
    # Convert single values to ranges for product generation
    ranges_for_product = []
    for key, value in ranges.items():
        if hasattr(value, '__iter__') and not isinstance(value, str):
            # It's already iterable (range, list, etc.)
            ranges_for_product.append(value)
        else:
            # It's a single value, convert to single-item range
            ranges_for_product.append([value])
    
    # Generate all combinations of the available indicators
    for values in product(*ranges_for_product):
        combo = {}
        
        # Create combo with only the indicators we're actually using
        for indicator_name, value in zip(indicator_names, values):
            combo[indicator_name] = value
        
        # Apply smart filtering rules
        skip_combo = False
        
        # Skip if EMA period >= VWMA period (only keep EMA < VWMA)
        if 'ema' in combo and 'vwma' in combo and combo['ema'] >= combo['vwma']:
            skip_combo = True
        
        # Skip if EMA fast >= EMA slow (only keep EMA fast < EMA slow)
        if 'ema_fast' in combo and 'ema_slow' in combo and combo['ema_fast'] >= combo['ema_slow']:
            skip_combo = True
        
        # Skip if MACD fast >= slow
        if 'macd_fast' in combo and 'macd_slow' in combo and combo['macd_fast'] >= combo['macd_slow']:
            skip_combo = True
        
        if not skip_combo:
            combinations.append(combo)
    logger.info(f"Generated {len(combinations)} combinations for indicators: {list(ranges.keys())} (filtered: EMA < VWMA, EMA_fast < EMA_slow, MACD_fast < MACD_slow)")
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
                # Remove timezone suffix (EST/EDT) and parse
                # e.g., "2025-01-02 09:30:00 EST" -> "2025-01-02 09:30:00"
                datetime_clean = data['datetime'].str.replace(r' E[SD]T$', '', regex=True)
                data['datetime'] = pd.to_datetime(datetime_clean, format='%Y-%m-%d %H:%M:%S')
            except:
                try:
                    # Fallback to automatic parsing
                    data['datetime'] = pd.to_datetime(data['datetime'].str.replace(r' E[SD]T$', '', regex=True))
                except:
                    # Final fallback
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

def save_trades_to_csv(trades: List[Dict], combo: Dict, symbol: str, timeframe: str, is_options: bool):
    """Save individual trades to CSV file"""
    try:
        # Create trades directory structure
        if is_options:
            trades_dir = f'data/trades/options/{symbol}'
        else:
            trades_dir = f'data/trades/{timeframe}/{symbol}'
        
        os.makedirs(trades_dir, exist_ok=True)
        
        # Create filename based on indicator parameters
        param_str = '_'.join([f"{k}_{v}" for k, v in combo.items()])
        filename = f'trades_{param_str}.csv'
        filepath = os.path.join(trades_dir, filename)
        
        # Convert trades to DataFrame and save
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(filepath, index=False)
        
        logger.debug(f"Saved {len(trades)} trades to {filepath}")
        
    except Exception as e:
        logger.warning(f"Failed to save trades for combo {combo}: {e}")

def process_chunk_optimized(args: Tuple) -> List[Dict]:
    """Optimized chunk processing function"""
    data, combinations_chunk, is_options, symbol, timeframe = args
    
    results = []
    for combo in combinations_chunk:
        try:
            # Ensure data is a copy to avoid issues in parallel processing
            data_copy = data.copy()
            result = backtest(data_copy, combo, is_options=is_options)
            
            # Save individual trades to CSV if trades exist
            if result.get('trades') and len(result['trades']) > 0:
                save_trades_to_csv(result['trades'], combo, symbol, timeframe, is_options)
            
            # Remove trades from result to keep summary data only
            result_without_trades = {k: v for k, v in result.items() if k != 'trades'}
            results.append(result_without_trades)
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
                future = executor.submit(process_chunk_optimized, (data, chunk, is_options, symbol, timeframe))
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
            chunk_results = process_chunk_optimized((data, chunk, is_options, symbol, timeframe))
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

    # Use indicator configuration from config file
    logger.info(f"Using indicator configuration: {list(INDICATOR_RANGES.keys())}")
    
    # Initialize overall progress tracker
    overall_tracker = ProgressTracker(len(symbols), "Overall Symbol Processing")
    
    for symbol in symbols:
        symbol_start = time.time()
        logger.info(f"Starting processing for {symbol}")
        
        # Generate combinations using the full indicator configuration
        test_combinations = generate_combinations_optimized(INDICATOR_RANGES)
        
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