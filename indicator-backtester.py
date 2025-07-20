import os
import sys
import pandas as pd

from backtester import backtest, run_single_backtest
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def test_roc_periods(base_indicator_periods):
    test_combinations = []
    for roc in range(3,21):
        for roc_of_roc in range(3,21):
            base_indicator_periods['roc'] = roc
            base_indicator_periods['roc_of_roc'] = roc_of_roc
            test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_ema_vwma_periods(base_indicator_periods):
    test_combinations = []
    for ema in range(3,21):
        for vwma in range(3,21):
            base_indicator_periods['ema'] = ema
            base_indicator_periods['vwma'] = vwma
            test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_macd_periods(base_indicator_periods):
    test_combinations = []
    for fast in range(2,42):
        for slow in range(2,42):
            for signal in range(2,42):
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
    test_combinations = []
    for period in range(2,20,2):
        for k in range(2,20,2):
            for d in range(2,20,2):
                base_indicator_periods['stoch_rsi_period'] = period
                base_indicator_periods['stoch_rsi_k'] = k
                base_indicator_periods['stoch_rsi_d'] = d
                test_combinations.append(base_indicator_periods.copy())
    return test_combinations

def test_indicator_period_combinations(data: pd.DataFrame, symbol: str, test_combinations: list[dict], use_parallel: bool = True, max_workers: int | None = None) -> pd.DataFrame:
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
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all backtests
            future_to_combo = {executor.submit(run_single_backtest, (data, combo)): combo for combo in test_combinations}
            
            # Collect results as they complete
            for future in as_completed(future_to_combo):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    combo = future_to_combo[future]
                    print(f'Backtest generated an exception: {exc}')
    else:
        # Sequential processing (original method)
        print(f"Running {len(test_combinations)} backtests sequentially...")
        results = []
        for indicator_periods in test_combinations:
            result = backtest(data, indicator_periods)
            results.append(result)
    # Save all results to CSV
    if results:
        result_df = pd.DataFrame(results)
        
        # Sort by total profit (descending)
        sorted_df = result_df.sort_values(['total_trade_profit'], ascending=[False])
        
        # Save sorted results
        sorted_df.to_csv(f'data/results/{symbol}.csv', mode='w', index=False)
        print(f"Saved {len(results)} results to data/results/{symbol}.csv (sorted by total profit)")
        
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

def main():
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
    # Testing different indicator period combinations
    for symbol in ['QQQ', 'SPY']:
        if symbol == 'QQQ':
            test_combinations = [qqq_base_indicator_periods]
        else:
            test_combinations = [spy_base_indicator_periods]
        data = pd.read_csv(f'data/5m/{symbol}.csv')
        test_indicator_period_combinations(data, symbol, test_combinations)

if __name__ == '__main__':
    main()