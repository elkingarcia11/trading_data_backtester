# Test different trade range combinations
# EX: Only open trades after 10:00am, 10:15am, 10:30am, 11am, 11:30am, 12pm, 12:30pm, 1pm, 1:30pm, 2pm, 2:30pm, 3pm, 3:30pm, 4pm
# EX: Close all trades by 2:30pm, 2:45pm, 3pm, 3:15pm, 3:30pm, 3:45pm, 4pm

import pandas as pd
from backtester import backtest

def test_trade_time_combinations():
    """
    Test different trade time combinations
    """

    # Load data
    data = pd.read_csv('data/5m/QQQ.csv')
    indicator_periods = {
        'ema': 7,
        'vwma': 6,
        'macd_fast': 21,
        'macd_slow': 37,
        'macd_signal': 15,
        'roc': 11,
        'roc_of_roc': 13,
    }

    data = pd.read_csv('data/5m/QQQ.csv')

    # Create a list of all possible start times from 9:30am to 3:50pm
    trade_times = []
    for hour in range(9, 16):
        for minute in range(0, 60, 5):
            trade_times.append(f'{hour}:{minute:02d}:00')
    
    for trade_time in trade_times:
        for trade_time_2 in trade_times:
            if trade_time_2 > trade_time:
                backtest(data, indicator_periods, trade_time, trade_time_2)

    pass