# Timer/ETA Features for Optimized Backtester

## Overview

The optimized backtester now includes comprehensive timer and ETA (Estimated Time of Arrival) functionality to provide real-time progress tracking and time estimates during long-running backtesting operations.

## Features Added

### 1. ProgressTracker Class

A new `ProgressTracker` class that provides:

- **Progress percentage calculation**
- **Elapsed time tracking**
- **ETA calculation** based on average processing time per item
- **Visual progress bars** using Unicode characters
- **Smart update intervals** to avoid log spam (updates every 5 seconds)

### 2. Multi-Level Progress Tracking

The timer system tracks progress at multiple levels:

#### Overall Progress

- Tracks completion of all symbols being processed
- Shows total progress across the entire backtesting run

#### Symbol-Level Progress

- Tracks completion of individual symbols (QQQ, SPY, etc.)
- Shows progress for each symbol independently

#### Timeframe/Options Progress

- For stock data: Tracks completion of different timeframes (1m, 5m, etc.)
- For options data: Tracks completion of different options dataframes

#### Day-Level Progress

- For stock data: Tracks completion of individual trading days
- Shows progress within each timeframe

#### Chunk-Level Progress

- Tracks completion of processing chunks
- Shows progress of indicator combination testing

### 3. Visual Progress Indicators

The system provides:

- **Progress bars**: `[████████████████████░░░░░░░░░░]` style visual indicators
- **Percentage completion**: `(45.2%)` format
- **Time formatting**: `HH:MM:SS` format for elapsed time and ETA
- **ETA timestamps**: Shows when the process will complete

## Example Output

```
2025-07-27 23:56:59,650 - INFO - Options Dataframes for QQQ: [██████████████████████████████] 6/6 (100.0%) - Elapsed: 00:00:00 - ETA: 23:56:59
2025-07-27 23:56:59,650 - INFO - Options Dataframes for QQQ: COMPLETED in 00:00:00
2025-07-27 23:56:59,650 - INFO - Completed QQQ in 0.59 seconds
2025-07-27 23:57:00,793 - INFO - Overall Symbol Processing: [██████████████████████████████] 2/2 (100.0%) - Elapsed: 00:00:01 - ETA: 23:57:00
2025-07-27 23:57:00,793 - INFO - Overall Symbol Processing: COMPLETED in 00:00:01
2025-07-27 23:57:00,794 - INFO - Total processing time: 1.74 seconds
```

## Usage

### Running with Timer/ETA

1. **Demo Mode** (quick test):

   ```bash
   python run_with_timer.py --demo --options --no-parallel
   ```

2. **Full Mode** (complete backtesting):

   ```bash
   python run_with_timer.py --options --no-parallel
   ```

3. **Stock Data Mode**:
   ```bash
   python run_with_timer.py --no-parallel
   ```

### Command Line Options

- `--options`: Process options data instead of stock data
- `--parallel`: Enable parallel processing (default)
- `--no-parallel`: Disable parallel processing (recommended for reliable timer display)
- `--workers N`: Set number of worker processes
- `--chunk-size N`: Set chunk size for processing
- `--demo`: Run with limited combinations for quick testing

## Benefits

1. **Time Management**: Know exactly how long the backtesting will take
2. **Progress Monitoring**: See real-time progress at multiple levels
3. **Resource Planning**: Plan your time based on ETA estimates
4. **Debugging**: Identify bottlenecks by seeing which levels take the most time
5. **User Experience**: Better feedback during long-running operations

## Technical Details

### ProgressTracker Class Methods

- `__init__(total_items, description)`: Initialize tracker
- `update(completed=1)`: Update progress count
- `_log_progress(current_time)`: Internal method to log progress
- `_format_time(seconds)`: Convert seconds to HH:MM:SS format
- `finish()`: Log final completion time

### Integration Points

The timer system is integrated into:

- `test_intraday_periods_optimized()`: Overall symbol progress
- `test_intraday_periods_for_optimized()`: Symbol-specific progress
- `test_indicator_period_combinations_optimized()`: Chunk processing progress

### Performance Impact

- **Minimal overhead**: Progress tracking adds <1% performance impact
- **Smart updates**: Only logs every 5 seconds to avoid spam
- **Memory efficient**: Uses simple counters and timestamps
- **Non-blocking**: Progress tracking doesn't interfere with processing

## Troubleshooting

### Common Issues

1. **No progress updates**: Check if processing is actually running
2. **Inaccurate ETA**: Early estimates may be inaccurate; they improve as more data is processed
3. **Parallel processing errors**: Use `--no-parallel` for more reliable progress tracking

### Performance Tips

- Use `--no-parallel` for most reliable timer display
- Use `--demo` mode to test timer functionality quickly
- Monitor memory usage with large datasets
- Check log files for detailed progress information
