import sys
import pandas as pd
sys.path.append('indicator-calculator')
from indicator_calculator import IndicatorCalculator
import datetime
import pytz

# Default signal thresholds configuration
DEFAULT_SIGNAL_THRESHOLDS = {
    'rsi': {'buy': 50, 'sell': 50},
    'stoch_rsi_k': {'buy': 50, 'sell': 50},
    'stoch_rsi_d': {'buy': 50, 'sell': 50},
    'macd': {'buy': 'line_above_signal', 'sell': 'line_below_signal'},
    'roc': {'buy': 0, 'sell': 0},
    'roc_of_roc': {'buy': 0, 'sell': 0},
    'ema_vwma': {'buy': 'ema_above_vwma', 'sell': 'ema_below_vwma'},
    'sma': {'buy': 0, 'sell': 0},
    'volatility': {'buy': 0, 'sell': 0},
    'price_change': {'buy': 0, 'sell': 0},
    'bollinger_bands': {'buy': 'price_between_bands', 'sell': 'price_outside_bands'},
    'bollinger_bands_width': {'buy': 0, 'sell': 0},
    'atr': {'buy': 0, 'sell': 0}
}

def run_single_backtest(args):
    """
    Helper function to run a single backtest for parallel processing
    """
    data, indicator_periods = args
    return backtest(data, indicator_periods)

def unix_to_edt_time(timestamp):
    # Convert milliseconds to seconds if timestamp is too large
    if timestamp > 1e12:  # If timestamp is in milliseconds (13+ digits)
        timestamp = timestamp / 1000
    
    utc_dt = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
    edt_tz = pytz.timezone('US/Eastern')
    edt_dt = utc_dt.astimezone(edt_tz)
    
    hour = edt_dt.hour  # 0-23
    minute = edt_dt.minute  # 0-59
    
    return hour, minute

def backtest(data: pd.DataFrame, indicator_periods: dict = {}, signal_thresholds: dict | None = None, start_time: tuple[int, int] | None = None, end_time: tuple[int, int] | None = None) -> dict:
    """
    Backtest the indicator combinations
    start time format int hours, int minutes
    end time format int hours, int minutes

    if trade open and current time is after end time exit trade, else check sell signals
    if trade not open and current time is < start time, do not open trade, else check buy signals
    """
    if signal_thresholds is None:
        signal_thresholds = DEFAULT_SIGNAL_THRESHOLDS.copy()
    
    data = IndicatorCalculator().calculate_all_indicators(data, indicator_periods)

    # Save dataframe with indicators to CSV for verification
    # data.to_csv(f'data/indicators/indicators_{indicator_periods}.csv', index=False)
    print(start_time, end_time)

    # Trades data
    trades = []

    # For each trade record the following:
    # Entry price, exit price, trade duration, profit, max unrealized profit, min unrealized profit
    trade_open = False
    entry_price = 0
    exit_price = 0
    trade_start_timestamp = 0
    max_unrealized_profit = 0
    min_unrealized_profit = float('inf')  # Initialize to infinity for proper min tracking

    trailing_stop_pct = 0.1
    stop_loss_pct = 0.05
    
    def should_exit_trade(row, trade_open, entry_price):
        """Helper function to determine if trade should be exited"""
        if not trade_open:
            return False
        
        # Check if current time is after end time
        if start_time and end_time:
            current_hour, current_minute = unix_to_edt_time(row['timestamp'])
            if current_hour > end_time[0] or (current_hour == end_time[0] and current_minute >= end_time[1]):
                return True
        
        # Check sell signals and stop conditions
        return (check_signal(row, 'sell', signal_thresholds) or 
                row['close'] <= entry_price * (1 - stop_loss_pct) or 
                row['close'] >= entry_price * (1 + trailing_stop_pct))
    
    def should_enter_trade(row):
        """Helper function to determine if trade should be entered"""
        # Check if current time is before start time
        if start_time and end_time:
            current_hour, current_minute = unix_to_edt_time(row['timestamp'])
            if current_hour < start_time[0] or (current_hour == start_time[0] and current_minute < start_time[1]):
                return False
            
            # Check if current time is before end time
            if current_hour > end_time[0] or (current_hour == end_time[0] and current_minute >= end_time[1]):
                return False
        
        return check_signal(row, 'buy', signal_thresholds)
    
    def close_trade(row):
        """Helper function to close a trade and record it"""
        nonlocal trade_open, entry_price, exit_price, trade_start_timestamp, max_unrealized_profit, min_unrealized_profit
        
        exit_price = row['close']
        trade_duration = row['timestamp'] - trade_start_timestamp
        profit = exit_price - entry_price
        
        trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'trade_duration': trade_duration,
            'profit': profit,
            'max_unrealized_profit': max_unrealized_profit,
            'min_unrealized_profit': min_unrealized_profit
        })

        # Reset trade state
        trade_open = False
        entry_price = 0
        exit_price = 0
        trade_start_timestamp = 0
        max_unrealized_profit = 0
        min_unrealized_profit = float('inf')
    
    for _, row in data.iterrows():
        if trade_open:
            # Update unrealized profit/loss
            current_pnl = row['close'] - entry_price
            max_unrealized_profit = max(max_unrealized_profit, current_pnl)
            min_unrealized_profit = min(min_unrealized_profit, current_pnl)
            
            # Check if we should exit the trade
            if should_exit_trade(row, trade_open, entry_price):
                close_trade(row)
        else:
            # Check if we should enter a trade
            if should_enter_trade(row):
                entry_price = row['close']
                trade_open = True
                trade_start_timestamp = row['timestamp']
                max_unrealized_profit = 0
                min_unrealized_profit = 0

    # Return the following: total_trades, total_trade_profit, average_trade_profit, win_rate, max_unrealized_profit, min_unrealized_profit, average_max_unrealized_profit, average_min_unrealized_profit, max_trade_duration, min_trade_duration, average_trade_duration, description_of_indicator_periods
    
    total_trades = len(trades)
    win_rate = 0 if len(trades) == 0 else len([trade for trade in trades if trade['profit'] > 0]) / len(trades)
    average_trade_profit = 0 if len(trades) == 0 else sum([trade['profit'] for trade in trades]) / len(trades)
    total_trade_profit = sum([trade['profit'] for trade in trades])
    max_unrealized_profit = max([trade['max_unrealized_profit'] for trade in trades]) if trades else 0
    min_unrealized_profit = min([trade['min_unrealized_profit'] for trade in trades]) if trades else 0
    average_max_unrealized_profit = 0 if len(trades) == 0 else sum([trade['max_unrealized_profit'] for trade in trades]) / len(trades)
    average_min_unrealized_profit = 0 if len(trades) == 0 else sum([trade['min_unrealized_profit'] for trade in trades]) / len(trades)
    max_trade_duration = max([trade['trade_duration'] for trade in trades]) if trades else 0
    min_trade_duration = min([trade['trade_duration'] for trade in trades]) if trades else 0
    average_trade_duration = 0 if len(trades) == 0 else sum([trade['trade_duration'] for trade in trades]) / len(trades)
    
    # Convert durations from milliseconds to minutes
    max_trade_duration = max_trade_duration / (1000 * 60)  # Convert ms to minutes
    min_trade_duration = min_trade_duration / (1000 * 60)  # Convert ms to minutes
    average_trade_duration = average_trade_duration / (1000 * 60)  # Convert ms to minutes
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'average_trade_profit': average_trade_profit,
        'total_trade_profit': total_trade_profit,
        'max_unrealized_profit': max_unrealized_profit,
        'min_unrealized_profit': min_unrealized_profit,
        'average_max_unrealized_profit': average_max_unrealized_profit,
        'average_min_unrealized_profit': average_min_unrealized_profit,
        'max_trade_duration (minutes)': max_trade_duration,
        'min_trade_duration (minutes)': min_trade_duration,
        'average_trade_duration (minutes)': average_trade_duration,
        'start_time': start_time,
        'end_time': end_time,
        **{f'{indicator}': indicator_periods[indicator] for indicator in indicator_periods},
    }

def check_signal(row: pd.Series, signal_type: str, signal_thresholds: dict) -> bool:
    """
    Unified function to check buy or sell signals based on available indicators
    """
    # Check for NaN/None in any value
    if row.isna().any():
        return False
    # Check for empty strings in any value
    if any(isinstance(v, str) and v.strip() == '' for v in row):
        return False
    # Check for None/empty/NaN in lists/tuples
    for v in row:
        if isinstance(v, (list, tuple)):
            if any(x is None or (isinstance(x, str) and x.strip() == '') or pd.isna(x) for x in v):
                return False
    conditions_met = 0
    conditions_to_be_met = 0
    # RSI check
    if 'rsi' in row:
        threshold = signal_thresholds['rsi'][signal_type]
        if (signal_type == 'buy' and row['rsi'] > threshold) or (signal_type == 'sell' and row['rsi'] < threshold):
            conditions_met += 1
        conditions_to_be_met += 1
    # Stochastic RSI K and Dcheck
    if 'stoch_rsi_k' in row and 'stoch_rsi_d' in row:
        if (signal_type == 'buy' and row['stoch_rsi_k'] > row['stoch_rsi_d']) or (signal_type == 'sell' and row['stoch_rsi_k'] < row['stoch_rsi_d']):
            conditions_met += 1
        conditions_to_be_met += 1
    # MACD check
    if 'macd_line' in row and 'macd_signal' in row:
        threshold_type = signal_thresholds['macd'][signal_type]
        if threshold_type == 'line_above_signal' and signal_type == 'buy' and row['macd_line'] > row['macd_signal']:
            conditions_met += 1
        elif threshold_type == 'line_below_signal' and signal_type == 'sell' and row['macd_line'] < row['macd_signal']:
            conditions_met += 1
        conditions_to_be_met += 1
    # ROC check
    if 'roc' in row:
        threshold = signal_thresholds['roc'][signal_type]
        if (signal_type == 'buy' and row['roc'] > threshold) or (signal_type == 'sell' and row['roc'] < threshold):
            conditions_met += 1
        conditions_to_be_met += 1
    # ROC of ROC check
    if 'roc_of_roc' in row:
        threshold = signal_thresholds['roc_of_roc'][signal_type]
        if (signal_type == 'buy' and row['roc_of_roc'] > threshold) or (signal_type == 'sell' and row['roc_of_roc'] < threshold):
            conditions_met += 1
        conditions_to_be_met += 1
    # EMA vs VWMA check
    if 'ema' in row and 'vwma' in row:
        threshold_type = signal_thresholds['ema_vwma'][signal_type]
        if threshold_type == 'ema_above_vwma' and signal_type == 'buy' and row['ema'] > row['vwma']:
            conditions_met += 1
        elif threshold_type == 'ema_below_vwma' and signal_type == 'sell' and row['ema'] < row['vwma']:
            conditions_met += 1
        conditions_to_be_met += 1
    # SMA check
    if 'sma' in row:
        threshold = signal_thresholds['sma'][signal_type]
        if (signal_type == 'buy' and row['sma'] > threshold) or (signal_type == 'sell' and row['sma'] < threshold):
            conditions_met += 1
        conditions_to_be_met += 1
    # Volatility check
    if 'volatility' in row:
        threshold = signal_thresholds['volatility'][signal_type]
        if (signal_type == 'buy' and row['volatility'] > threshold) or (signal_type == 'sell' and row['volatility'] < threshold):
            conditions_met += 1
        conditions_to_be_met += 1
    # Price change check
    if 'price_change' in row:
        threshold = signal_thresholds['price_change'][signal_type]
        if (signal_type == 'buy' and row['price_change'] > threshold) or (signal_type == 'sell' and row['price_change'] < threshold):
            conditions_met += 1
        conditions_to_be_met += 1
    # Bollinger Bands check (FIXED LOGIC)
    if 'bollinger_bands' in row:
        threshold_type = signal_thresholds['bollinger_bands'][signal_type]
        bb_lower, bb_upper = row['bollinger_bands']
        price = row['close']
        if threshold_type == 'price_between_bands' and signal_type == 'buy' and bb_lower < price < bb_upper:
            conditions_met += 1
        elif threshold_type == 'price_outside_bands' and signal_type == 'sell' and (price < bb_lower or price > bb_upper):
            conditions_met += 1  
        conditions_to_be_met += 1
    # Bollinger Bands width check
    if 'bollinger_bands_width' in row:
        threshold = signal_thresholds['bollinger_bands_width'][signal_type]
        if (signal_type == 'buy' and row['bollinger_bands_width'] > threshold) or (signal_type == 'sell' and row['bollinger_bands_width'] < threshold):
            conditions_met += 1
        conditions_to_be_met += 1
    # ATR check
    if 'atr' in row:
        threshold = signal_thresholds['atr'][signal_type]
        if (signal_type == 'buy' and row['atr'] > threshold) or (signal_type == 'sell' and row['atr'] < threshold):
            conditions_met += 1
        conditions_to_be_met += 1
    return conditions_met >= conditions_to_be_met if signal_type == 'buy' else conditions_met >= conditions_to_be_met - 1