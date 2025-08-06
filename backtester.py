import pytz
import sys
import pandas as pd
sys.path.append('indicator-calculator')
from indicator_calculator import IndicatorCalculator

def run_single_backtest(args):
    """
    Helper function to run a single backtest for parallel processing
    """
    data, indicator_periods, is_options = args
    
    return backtest(data, indicator_periods, is_options=is_options)


def unix_to_edt_time(timestamp):
    """Convert timestamp to EDT hour and minute, handling both Unix timestamps and datetime strings"""
    import datetime as dt
    
    # Handle datetime strings (like "2025-07-24 09:30:03")
    if isinstance(timestamp, str):
        try:
            # Parse datetime string
            dt_obj = dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            # Assume it's already in EDT/EST timezone
            hour = dt_obj.hour
            minute = dt_obj.minute
            return hour, minute
        except ValueError:
            # Try alternative format if the first one fails
            try:
                dt_obj = dt.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S %Z')
                hour = dt_obj.hour
                minute = dt_obj.minute
                return hour, minute
            except ValueError:
                print(f"Warning: Could not parse timestamp string: {timestamp}")
                return 9, 30  # Default to market open time
    
    # Handle numeric Unix timestamps
    else:
        # Convert milliseconds to seconds if timestamp is too large
        if timestamp > 1e12:  # If timestamp is in milliseconds (13+ digits)
            timestamp = timestamp / 1000

        utc_dt = dt.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        edt_tz = pytz.timezone('US/Eastern')
        edt_dt = utc_dt.astimezone(edt_tz)

        hour = edt_dt.hour  # 0-23
        minute = edt_dt.minute  # 0-59

        return hour, minute


def calculate_trade_duration(start_timestamp, end_timestamp):
    """Calculate duration between two timestamps, handling both Unix timestamps and datetime strings"""
    import datetime as dt
    
    # Handle datetime strings
    if isinstance(start_timestamp, str) and isinstance(end_timestamp, str):
        try:
            start_dt = dt.datetime.strptime(start_timestamp, '%Y-%m-%d %H:%M:%S')
            end_dt = dt.datetime.strptime(end_timestamp, '%Y-%m-%d %H:%M:%S')
            duration = end_dt - start_dt
            # Convert to milliseconds for consistency
            return duration.total_seconds() * 1000
        except ValueError:
            # Try alternative format
            try:
                start_dt = dt.datetime.strptime(start_timestamp, '%Y-%m-%d %H:%M:%S %Z')
                end_dt = dt.datetime.strptime(end_timestamp, '%Y-%m-%d %H:%M:%S %Z')
                duration = end_dt - start_dt
                return duration.total_seconds() * 1000
            except ValueError:
                print(f"Warning: Could not parse timestamp strings for duration calculation")
                return 0
    
    # Handle datetime objects (pandas Timestamp)
    elif hasattr(start_timestamp, 'timestamp') and hasattr(end_timestamp, 'timestamp'):
        duration = end_timestamp - start_timestamp
        # Convert to milliseconds for consistency
        return duration.total_seconds() * 1000
    
    # Handle numeric Unix timestamps
    else:
        return end_timestamp - start_timestamp


def backtest(data: pd.DataFrame, indicator_periods: dict = {}, start_time: tuple[int, int] | None = None, end_time: tuple[int, int] | None = None, is_options: bool = False) -> dict:
    """
    Backtest the indicator combinations
    start time format int hours, int minutes
    end time format int hours, int minutes

    if trade open and current time is after end time exit trade, else check sell signals
    if trade not open and current time is < start time, do not open trade, else check buy signals
    """
    try:
        # Determine the correct price column based on whether it's options data
        price_column = 'last_price' if is_options else 'close'
        data = IndicatorCalculator().calculate_all_indicators(data, indicator_periods, price_column)
        # Save dataframe with indicators to CSV for verification
        #data.to_csv(f'data/indicators/indicators_{indicator_periods}.csv', index=False)
    except Exception as e:
        print(f"Error in calculate_all_indicators: {e}")
        print(f"Indicator periods: {indicator_periods}")
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"First few rows of problematic columns:")
        price_cols = ['last_price', 'close', 'volume']
        for col in price_cols:
            if col in data.columns:
                print(f"  {col}: {data[col].head().tolist()}")
        raise e

    # Trades data
    trades = []

    # For each trade record the following:
    # Entry price, exit price, trade duration, profit, max unrealized profit, min unrealized profit
    trade_open = False
    entry_price = 0
    exit_price = 0
    trade_start_timestamp = 0
    max_price_seen = entry_price
    min_price_seen = entry_price

    # Trailing stop is highest last price * 0.9
    trailing_stop_pct = 0.9
    stop_loss_pct = 0.95

    def should_exit_trade(row, trade_open, entry_price, is_options: bool = False):
        """Helper function to determine if trade should be exited"""
        if not trade_open:
            return False

        # Check if current time is after end time
        if start_time and end_time:
            current_hour, current_minute = unix_to_edt_time(row['timestamp'])
            if current_hour > end_time[0] or (current_hour == end_time[0] and current_minute >= end_time[1]):
                return True

        # Check sell signals and stop conditions
        if check_signal_combos(row, 'sell'):
            return True
            
        # Get current price based on asset type  
        current_price = row['last_price'] if is_options else row['close']
        
        # Check stop loss and trailing stop conditions
        return (current_price <= entry_price * stop_loss_pct or 
                current_price <= max_price_seen * trailing_stop_pct)

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

        return check_signal_combos(row, 'buy')

    def close_trade(row, entry_indicator_values, exit_indicator_values):
        """Helper function to close a trade and record it"""
        nonlocal trade_open, entry_price, exit_price, trade_start_timestamp, max_price_seen, min_price_seen

        try:
            exit_price = row['close'] if not is_options else row['last_price']
            trade_duration = calculate_trade_duration(trade_start_timestamp, row['timestamp'])
            profit = exit_price - entry_price
            max_unrealized_profit = max_price_seen - entry_price
            max_drawdown = min_price_seen - entry_price

            trades.append({
                'entry_price': entry_price,
                'entry_timestamp': trade_start_timestamp,
                'exit_price': exit_price,
                'exit_timestamp': row['timestamp'],
                'trade_duration': trade_duration,
                'profit': profit,
                'max_unrealized_profit': max_unrealized_profit,
                'max_drawdown': max_drawdown,
                **{f'entry_{indicator}': entry_indicator_values[indicator] for indicator in entry_indicator_values},
                **{f'exit_{indicator}': exit_indicator_values[indicator] for indicator in exit_indicator_values},
            })
        except Exception as e:
            print(f"🔍 String error in close_trade:")
            print(f"  exit_price type: {type(exit_price)}, value: {exit_price}")
            print(f"  entry_price type: {type(entry_price)}, value: {entry_price}")
            print(f"  max_price_seen type: {type(max_price_seen)}, value: {max_price_seen}")
            print(f"  min_price_seen type: {type(min_price_seen)}, value: {min_price_seen}")
            print(f"  row['close'] type: {type(row.get('close', 'N/A'))}, value: {row.get('close', 'N/A')}")
            print(f"  row['last_price'] type: {type(row.get('last_price', 'N/A'))}, value: {row.get('last_price', 'N/A')}")
            raise e

        # Reset trade state
        trade_open = False
        entry_price = 0
        exit_price = 0
        trade_start_timestamp = 0
        max_price_seen = entry_price
        min_price_seen = entry_price

    # Counters
    total_rows = 0
    buy_signal_count = 0
    sell_signal_count = 0
    
    for _, row in data.iterrows():
        total_rows += 1
        if total_rows < indicator_periods['macd_slow'] * 3:
            continue
        elif trade_open:
                # Update unrealized profit/loss
                current_price = row['close'] if not is_options else row['last_price']
                max_price_seen = max(max_price_seen, current_price)
                min_price_seen = min(min_price_seen, current_price)

                # Check if we should exit the trade
                if should_exit_trade(row, trade_open, entry_price, is_options):
                    exit_indicator_values = {
                        'ema': row.get('ema'),
                        'vwma': row.get('vwma'),
                        'macd_line': row.get('macd_line'),
                        'macd_signal': row.get('macd_signal'),
                        'stoch_rsi_k': row.get('stoch_rsi_k'),
                        'stoch_rsi_d': row.get('stoch_rsi_d'),
                        'roc': row.get('roc'),
                        'roc_of_roc': row.get('roc_of_roc')
                    }
                    close_trade(row, entry_indicator_values, exit_indicator_values)
                    sell_signal_count += 1
        else:
            # Check if we should enter a trade
            if should_enter_trade(row):
                entry_price = row['close'] if not is_options else row['last_price']
                entry_indicator_values = {
                    'ema': row.get('ema'),
                    'vwma': row.get('vwma'),
                    'macd_line': row.get('macd_line'),
                    'macd_signal': row.get('macd_signal'),
                    'stoch_rsi_k': row.get('stoch_rsi_k'),
                    'stoch_rsi_d': row.get('stoch_rsi_d'),
                    'roc': row.get('roc'),
                    'roc_of_roc': row.get('roc_of_roc')
                }
                trade_open = True
                trade_start_timestamp = row['timestamp']
                max_price_seen = entry_price
                min_price_seen = entry_price
                buy_signal_count += 1

    # Return the following: total_trades, total_trade_profit, average_trade_profit, win_rate, max_unrealized_profit, min_unrealized_profit, average_max_unrealized_profit, average_min_unrealized_profit, max_trade_duration, min_trade_duration, average_trade_duration, description_of_indicator_periods

    total_trades = len(trades)
    win_rate = 0 if len(trades) == 0 else len(
        [trade for trade in trades if trade['profit'] > 0]) / len(trades)
    average_trade_profit = 0 if len(trades) == 0 else sum(
        [trade['profit'] for trade in trades]) / len(trades)
    # SUM all profit and divde by sum of all entry prices
    average_trade_profit_percentage = 0 if len(trades) == 0 else sum(
        [trade['profit'] for trade in trades]) / sum(
        [trade['entry_price'] for trade in trades])
    total_trade_profit = sum([trade['profit'] for trade in trades])
    max_unrealized_profit = max(
        [trade['max_unrealized_profit'] for trade in trades]) if trades else 0
    max_drawdown = min(
        [trade['max_drawdown'] for trade in trades]) if trades else 0
    average_max_unrealized_profit = 0 if len(trades) == 0 else sum(
        [trade['max_unrealized_profit'] for trade in trades]) / len(trades)
    average_max_drawdown = 0 if len(trades) == 0 else sum(
        [trade['max_drawdown'] for trade in trades]) / len(trades)
    max_trade_duration = max([trade['trade_duration']
                             for trade in trades]) if trades else 0
    min_trade_duration = min([trade['trade_duration']
                             for trade in trades]) if trades else 0
    average_trade_duration = 0 if len(trades) == 0 else sum(
        [trade['trade_duration'] for trade in trades]) / len(trades)

    # Convert durations from milliseconds to minutes
    max_trade_duration = max_trade_duration / \
        (1000 * 60)  # Convert ms to minutes
    min_trade_duration = min_trade_duration / \
        (1000 * 60)  # Convert ms to minutes
    average_trade_duration = average_trade_duration / \
        (1000 * 60)  # Convert ms to minutes

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'winning_trades': total_trades * win_rate,
        'average_trade_profit': average_trade_profit,
        'average_trade_profit_percentage': average_trade_profit_percentage,
        'expected_profit': win_rate*average_trade_profit_percentage*100,
        'total_trade_profit': total_trade_profit,
        'max_unrealized_profit': max_unrealized_profit,
        'max_drawdown': max_drawdown,
        'average_max_unrealized_profit': average_max_unrealized_profit,
        'average_max_drawdown': average_max_drawdown,
        'max_trade_duration (minutes)': max_trade_duration,
        'min_trade_duration (minutes)': min_trade_duration,
        'average_trade_duration (minutes)': average_trade_duration,
        'start_time': start_time,
        'end_time': end_time,
        'trades': trades,  # Add the detailed trades data
        **{f'{indicator}': indicator_periods[indicator] for indicator in indicator_periods},
    }
    
def check_signal_combos(row: pd.Series, signal_type: str) -> bool:
    """
    Check if any of the signal combos are met
    """
    
    if signal_type == 'buy':
        trend_conditions_met = 0
        trend_conditions_to_be_met = 2
        momentum_conditions_met = 0
        momentum_conditions_to_be_met = 3

        if 'ema' in row and 'vwma' in row:
            # Check if both are not nan
            if pd.isna(row['ema']) or pd.isna(row['vwma']):
                return False
            elif row['ema'] > row['vwma']:
                trend_conditions_met += 1

        if 'macd_line' in row and 'macd_signal' in row:
            if pd.isna(row['macd_line']) or pd.isna(row['macd_signal']):
                return False
            elif row['macd_line'] > row['macd_signal']:
                trend_conditions_met += 1

        if 'stoch_rsi_k' in row and 'stoch_rsi_d' in row:
            if pd.isna(row['stoch_rsi_k']) or pd.isna(row['stoch_rsi_d']):
                return False
            elif row['stoch_rsi_k'] > row['stoch_rsi_d']:
                momentum_conditions_met += 1

        if 'roc' in row:
            if pd.isna(row['roc']):
                return False
            elif row['roc'] > 0:
                momentum_conditions_met += 1
                
        if 'roc_of_roc' in row:
            if pd.isna(row['roc_of_roc']):
                return False
            elif row['roc_of_roc'] > 0:
                momentum_conditions_met += 1

    elif signal_type == 'sell':
        trend_conditions_met = 0
        trend_conditions_to_be_met = 2
        momentum_conditions_met = 0
        momentum_conditions_to_be_met = 10
        if 'ema' in row and 'vwma' in row:
            if pd.isna(row['ema']) or pd.isna(row['vwma']):
                return False
            elif row['ema'] < row['vwma']:
                trend_conditions_met += 1

        if 'macd_line' in row and 'macd_signal' in row:
            if pd.isna(row['macd_line']) or pd.isna(row['macd_signal']):
                return False
            elif row['macd_line'] < row['macd_signal']:
                trend_conditions_met += 1

        if 'stoch_rsi_k' in row and 'stoch_rsi_d' in row:
            if pd.isna(row['stoch_rsi_k']) or pd.isna(row['stoch_rsi_d']):
                return False
            if row['stoch_rsi_k'] < row['stoch_rsi_d']:
                momentum_conditions_met += 1

        if 'roc' in row:
            if pd.isna(row['roc']):
                return False
            elif row['roc'] < 0:  
                momentum_conditions_met += 1

        if 'roc_of_roc' in row:
            if pd.isna(row['roc_of_roc']):
                return False
            elif row['roc_of_roc'] < 0:
                momentum_conditions_met += 1

    return trend_conditions_met >= trend_conditions_to_be_met and momentum_conditions_met >= momentum_conditions_to_be_met