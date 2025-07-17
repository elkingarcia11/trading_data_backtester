# Trading Data Backtester

A flexible and extensible backtesting framework for evaluating trading strategies based on technical indicators. This project allows you to test various indicator combinations and parameter settings on historical price data, with support for parallelized backtesting for faster results.

## Features

- **Multiple Technical Indicators:** Supports RSI, MACD, ROC, EMA, VWMA, SMA, Bollinger Bands, ATR, and more.
- **Configurable Signal Logic:** Easily adjust buy/sell thresholds and indicator parameters.
- **Parallelized Backtesting:** Run multiple backtests simultaneously using all available CPU cores.
- **Comprehensive Trade Metrics:** Tracks entry/exit, profit, duration, win rate, and more.
- **Results Export:** Outputs results to CSV for further analysis.

## Installation

1. **Clone the repository (including submodules):**

   ```bash
   git clone --recursive <your-repo-url>
   cd trading_data_backtester
   ```

   _Note: The `--recursive` flag ensures that any submodules are also cloned._

2. **(Recommended) Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r indicator_calculator/requirements.txt
   ```

## Usage

1. **Prepare your data:**

   - Place your historical OHLCV data (CSV format) in the `data/` directory (e.g., `data/5m/SPY.csv`).
   - The CSV should include columns like `timestamp`, `open`, `high`, `low`, `close`, `volume`.

2. **Run the backtester:**

   ```bash
   python3 indicator_backtester.py
   ```

   By default, this will run `test_indicator_period_combinations` in parallel mode, testing different indicator period settings and saving results to `results.csv`.

3. **Customize parameters:**
   - Edit `indicator_backtester.py` to change indicator periods, thresholds, or which indicators are tested.
   - You can also call `test_indicator_period_combinations(use_parallel=False)` to run sequentially.

## Parallelization

- The backtester uses Python's `concurrent.futures.ProcessPoolExecutor` to run multiple backtests in parallel.
- By default, it uses all available CPU cores. You can limit the number of workers by passing `max_workers` to `test_indicator_period_combinations`.

## Requirements

- Python 3.8+
- See `indicator_calculator/requirements.txt` for required Python packages (e.g., pandas, numpy).

## Project Structure

```
trading_data_backtester/
├── indicator_backtester.py         # Main backtesting script
├── trade_time_backtester.py       # (Optional) Additional backtesting logic
├── indicator_calculator/
│   ├── indicator_calculator.py    # Technical indicator calculations
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Indicator calculator details
└── data/                          # Place your historical data here
```

## Extending

- Add new indicators in `indicator_calculator/indicator_calculator.py`.
- Adjust or add new backtest logic in `indicator_backtester.py`.

## Contact

For questions or contributions, please open an issue or contact the maintainer.
