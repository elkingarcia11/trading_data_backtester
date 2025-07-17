import os
import sys
import pandas as pd

from backtester import backtest, run_single_backtest
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def test_indicator_combinations() -> pd.DataFrame:
    """
    TODO: Test different indicator combinations
    """
    pass

def test_indicator_period_combinations(use_parallel: bool = True, max_workers: int | None = None) -> pd.DataFrame:
    """
    Test different indicator period combinations with optional parallelization
    
    Args:
        use_parallel: Whether to use parallel processing
        max_workers: Maximum number of worker processes (defaults to CPU count)
    """
    # Load data
    data = pd.read_csv('data/5m/QQQ.csv')
    base_indicator_periods = {
        'ema': 7,
        'vwma': 6,
        'macd_fast': 21,
        'macd_slow': 37,
        'macd_signal': 15,
        'roc': 11,
        'roc_of_roc': 10,
    }

    # Prepare all combinations to test
    test_combinations = []
    
    # Add base case with no ROC of ROC
    base_case_periods = base_indicator_periods.copy()
    if 'roc_of_roc' in base_case_periods:
        del base_case_periods['roc_of_roc']
    test_combinations.append((data, base_case_periods))
    
    # Add combinations with different ROC of ROC periods
    for i in range(1, 20):
        indicator_periods = base_indicator_periods.copy()
        indicator_periods['roc_of_roc'] = i
        test_combinations.append((data, indicator_periods))

    if use_parallel:
        # Use parallel processing
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()
        
        print(f"Running {len(test_combinations)} backtests using {max_workers} parallel workers...")
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all backtests
            future_to_combo = {executor.submit(run_single_backtest, combo): combo for combo in test_combinations}
            
            # Collect results as they complete
            for future in as_completed(future_to_combo):
                try:
                    result = future.result()
                    results.append(result)
                    roc_period = result['description_of_indicator_periods'].get('roc_of_roc', 'None')
                    print(f"Completed backtest for roc_of_roc={roc_period}")
                except Exception as exc:
                    combo = future_to_combo[future]
                    print(f'Backtest generated an exception: {exc}')
    else:
        # Sequential processing (original method)
        print(f"Running {len(test_combinations)} backtests sequentially...")
        results = []
        for data, indicator_periods in test_combinations:
            result = backtest(data, indicator_periods)
            results.append(result)
            roc_period = result['description_of_indicator_periods'].get('roc_of_roc', 'None')
            print(f"Completed backtest for roc_of_roc={roc_period}")
    # Save all results to CSV
    if results:
        result_df = pd.DataFrame(results)
        
        # Sort by win rate (descending) then by average trade profit (descending)
        sorted_df = result_df.sort_values(['win_rate', 'average_trade_profit'], ascending=[False, False])
        
        # Save sorted results
        sorted_df.to_csv('results.csv', mode='w', index=False)
        print(f"Saved {len(results)} results to results.csv (sorted by win rate then avg trade profit)")
        
        # Display top 3 performers
        print("\n=== TOP 3 PERFORMERS ===")
        for idx, row in sorted_df.head(3).iterrows():
            roc_period = row['description_of_indicator_periods'].get('roc_of_roc', 'None')
            win_rate = row['win_rate']
            avg_profit = row['average_trade_profit']
            total_trades = row['total_trades']
            print(f"{idx + 1}. ROC of ROC Period {roc_period}: Win Rate {win_rate:.2%}, "
                  f"Avg Profit ${avg_profit:.4f}, Total Trades {total_trades}")
        
        return sorted_df
    return pd.DataFrame() if results else pd.DataFrame()

def main():
   test_indicator_period_combinations()

if __name__ == '__main__':
    main()