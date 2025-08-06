import pandas as pd
import numpy as np
import os
import sys
sys.path.append('indicator-calculator')
from indicator_calculator import IndicatorCalculator
import indicator_config

def find_optimal_trades(df, number_of_trades):
    """
    Find optimal buy/sell trades to maximize profit with limited number of trades.
    
    Args:
        df (pd.DataFrame): DataFrame with 'last_price' column
        number_of_trades (int): Maximum number of complete buy-sell transactions
    
    Returns:
        pd.DataFrame: Original DataFrame with added 'action' column ('buy', 'sell', or 'hold')
    """
    # Make a copy to avoid modifying original dataframe
    result_df = df.copy()
    n = len(df)
    prices = df['last_price'].values
    
    # Initialize action column
    result_df['action'] = 'hold'
    
    # Edge cases
    if n <= 1 or number_of_trades <= 0:
        return result_df
    
    # Dynamic programming approach for limited trades
    # dp[i][j][k] = max profit at day i, with j transactions, k=0 (no stock), k=1 (holding stock)
    # We'll use a more memory-efficient approach
    
    # buy[i][j] = max profit after at most j transactions, currently holding stock, on day i
    # sell[i][j] = max profit after at most j transactions, not holding stock, on day i
    
    buy = [[-float('inf')] * (number_of_trades + 1) for _ in range(n)]
    sell = [[0] * (number_of_trades + 1) for _ in range(n)]
    
    # Initialize first day
    for j in range(number_of_trades + 1):
        buy[0][j] = -prices[0]  # Buy on first day
        sell[0][j] = 0  # Don't have stock
    
    # Fill DP table
    for i in range(1, n):
        for j in range(number_of_trades + 1):
            # sell[i][j]: either we already didn't have stock, or we sell today
            sell[i][j] = max(sell[i-1][j], buy[i-1][j] + prices[i])
            
            # buy[i][j]: either we already had stock, or we buy today (if we have transactions left)
            if j > 0:
                buy[i][j] = max(buy[i-1][j], sell[i-1][j-1] - prices[i])
            else:
                buy[i][j] = buy[i-1][j]
    
    # Backtrack to find actual trades
    _backtrack_trades(result_df, prices, buy, sell, number_of_trades)
    
    return result_df

def _backtrack_trades(df, prices, buy, sell, max_trades):
    """Backtrack through DP solution to find actual buy/sell points."""
    n = len(prices)
    i = n - 1
    j = max_trades
    holding = False
    
    # Determine if we end with stock or not
    if buy[i][j] + prices[i] > sell[i][j]:
        holding = True
        df.loc[i, 'action'] = 'sell'  # We should sell at the end
    
    while i > 0 and j > 0:
        if holding:
            # We're holding stock, check if we bought today
            if buy[i][j] != buy[i-1][j]:
                # We bought today
                df.loc[i, 'action'] = 'buy'
                holding = False
                j -= 1
        else:
            # We're not holding stock, check if we sold today
            if j > 0 and sell[i][j] != sell[i-1][j]:
                # We sold today, so we were holding yesterday
                df.loc[i, 'action'] = 'sell'
                holding = True
        i -= 1
    
    # Check first day
    if holding and i == 0:
        df.loc[0, 'action'] = 'buy'

def find_percent_gain_trades(df, percent_gain=5.0):
    """
    Find trades that achieve at least the specified percentage gain.
    Maximizes the number of trades that meet the minimum gain threshold.
    
    Args:
        df (pd.DataFrame): DataFrame with 'last_price' column
        percent_gain (float): Minimum percentage gain required (default 5.0%)
    
    Returns:
        pd.DataFrame: Original DataFrame with added 'action' column ('buy', 'sell', or 'hold')
    """
    # Make a copy to avoid modifying original dataframe
    result_df = df.copy()
    n = len(df)
    prices = df['last_price'].values
    
    # Initialize action column
    result_df['action'] = 'hold'
    
    # Edge cases
    if n <= 1:
        return result_df
    
    # Convert percentage to decimal
    min_gain_ratio = 1 + (percent_gain / 100.0)
    
    # Greedy approach: find local minima and maxima that satisfy gain requirement
    i = 0
    holding = False
    buy_price = 0
    buy_index = 0
    
    while i < n:
        if not holding:
            # Look for a good buying opportunity (local minimum)
            # Check if there's a future price that gives us the required gain
            best_future_gain = 0
            best_future_index = -1
            
            for j in range(i + 1, n):
                potential_gain = prices[j] / prices[i]
                if potential_gain >= min_gain_ratio and potential_gain > best_future_gain:
                    best_future_gain = potential_gain
                    best_future_index = j
            
            if best_future_index != -1:
                # We found a profitable trade, buy now
                result_df.loc[i, 'action'] = 'buy'
                holding = True
                buy_price = prices[i]
                buy_index = i
                
                # Now find the optimal sell point
                best_sell_price = 0
                best_sell_index = -1
                
                for j in range(i + 1, n):
                    if prices[j] >= buy_price * min_gain_ratio:
                        # This meets our minimum gain requirement
                        # Check if this is a local peak or if we should wait
                        is_good_sell = True
                        
                        # Look ahead to see if price will continue rising significantly
                        for k in range(j + 1, min(j + 3, n)):  # Look 2-3 days ahead
                            if prices[k] > prices[j] * 1.02:  # If price rises >2% more, wait
                                is_good_sell = False
                                break
                        
                        if is_good_sell and prices[j] > best_sell_price:
                            best_sell_price = prices[j]
                            best_sell_index = j
                
                if best_sell_index != -1:
                    result_df.loc[best_sell_index, 'action'] = 'sell'
                    holding = False
                    i = best_sell_index + 1
                else:
                    # No good sell point found, cancel the buy
                    result_df.loc[i, 'action'] = 'hold'
                    holding = False
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return result_df

def find_percent_gain_trades_simple(df, percent_gain=5.0):
    """
    Simpler approach: find all non-overlapping trades with at least percent_gain.
    Uses a greedy approach to maximize number of qualifying trades.
    
    Args:
        df (pd.DataFrame): DataFrame with 'last_price' column  
        percent_gain (float): Minimum percentage gain required (default 5.0%)
    
    Returns:
        pd.DataFrame: Original DataFrame with added 'action' column
    """
    result_df = df.copy()
    n = len(df)
    prices = df['last_price'].values
    
    result_df['action'] = 'hold'
    
    if n <= 1:
        return result_df
    
    min_gain_ratio = 1 + (percent_gain / 100.0)
    
    i = 0
    while i < n - 1:
        # Find next local minimum
        while i < n - 1 and prices[i] >= prices[i + 1]:
            i += 1
        
        if i >= n - 1:
            break
            
        buy_price = prices[i]
        buy_index = i
        
        # Look for sell opportunity with required gain
        j = i + 1
        best_sell_index = -1
        
        while j < n:
            if prices[j] >= buy_price * min_gain_ratio:
                # Found a price that meets our gain requirement
                # Look for local maximum from here
                best_sell_index = j
                while j < n - 1 and prices[j] <= prices[j + 1]:
                    j += 1
                    if prices[j] >= buy_price * min_gain_ratio:
                        best_sell_index = j
                break
            j += 1
        
        if best_sell_index != -1:
            # Execute the trade
            result_df.loc[buy_index, 'action'] = 'buy'
            result_df.loc[best_sell_index, 'action'] = 'sell'
            i = best_sell_index + 1
        else:
            i += 1
    
    return result_df


if __name__ == "__main__":
    # Load dfs from the data/options directory
    for file in os.listdir('data/options'):
        if file.endswith('.csv'):
            df = pd.read_csv(f'data/options/{file}')
            # Find best buy/sell rows to maximize profit using each columns 'last_price' as the price to buy/sell, with number_of_trades trades, add a column to record when we buy or sell
            # Export the results to a new csv file
            # Generate indicator for df
            calculator = IndicatorCalculator()
            df = calculator.calculate_all_indicators(data=df, indicator_periods=indicator_config.CUSTOM_CONFIG, is_option=True)
            # Return the results
            df = find_percent_gain_trades(df, 5)
            # Export the results to a new csv file
            df.to_csv(f'data/options/optimized_{file}', index=False)