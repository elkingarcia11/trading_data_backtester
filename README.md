# Trading Data Backtester

A flexible and extensible backtesting framework for evaluating trading strategies based on technical indicators. This project allows you to test various indicator combinations and parameter settings on historical price data, with support for parallelized backtesting for faster results.

## Features

- **Multiple Technical Indicators:** Supports RSI, MACD, ROC, EMA, VWMA, SMA, Bollinger Bands, ATR, and more.
- **Configurable Signal Logic:** Easily adjust buy/sell thresholds and indicator parameters.
- **Parallelized Backtesting:** Run multiple backtests simultaneously using all available CPU cores.
- **Comprehensive Trade Metrics:** Tracks entry/exit, profit, duration, win rate, and more.
- **Results Export:** Outputs results to CSV for further analysis.
- **Integrated Submodules:**
  - [indicator-calculator](indicator-calculator/README.md): Calculate technical indicators on financial data.
  - [market-data-api](market-data-api/README.md): Fetch historical market data from the Schwab Market Data API.

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
3. **Install dependencies for submodules:**
   ```bash
   pip install -r indicator-calculator/requirements.txt
   pip install -r market-data-api/requirements.txt
   ```

## Usage

1. **Prepare your data:**

   - Place your historical OHLCV data (CSV format) in the `data/` directory (e.g., `data/5m/SPY.csv`).
   - The CSV should include columns like `timestamp`, `open`, `high`, `low`, `close`, `volume`.
   - Alternatively, use the [market-data-api](market-data-api/README.md) submodule to fetch data from the Schwab Market Data API.

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
- See `indicator-calculator/requirements.txt` and `market-data-api/requirements.txt` for required Python packages (e.g., pandas, numpy, requests).

## Project Structure

```
trading_data_backtester/
├── indicator_backtester.py         # Main backtesting script
├── trade_time_backtester.py       # (Optional) Additional backtesting logic
├── indicator-calculator/          # Submodule: Technical indicator calculations
│   ├── indicator_calculator.py    # Technical indicator calculations
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Indicator calculator details
├── market-data-api/               # Submodule: Schwab Market Data API client
│   ├── schwab_market_data_client.py # API client for fetching historical data
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Market data API usage and details
└── data/                          # Place your historical data here
```

## Submodules

### indicator-calculator

A Python module for calculating various technical indicators on financial market data. See [indicator-calculator/README.md](indicator-calculator/README.md) for details and usage examples.

### market-data-api

A robust Python client for fetching historical market data from the Schwab Market Data API with optimal period chunking and comprehensive data quality validation. See [market-data-api/README.md](market-data-api/README.md) for setup and usage.

## Extending

- Add new indicators in `indicator-calculator/indicator_calculator.py`.
- Adjust or add new backtest logic in `indicator_backtester.py`.
- Use or extend the Schwab Market Data API client in `market-data-api/schwab_market_data_client.py`.

## Contact

For questions or contributions, please open an issue or contact the maintainer.
