#!/usr/bin/env python3
"""
Comprehensive indicator plotting script
Creates a multi-panel chart with:
1. Price, EMA, and VWMA
2. ROC and ROC of ROC (overlapping)
3. MACD line, signal line, and histogram
4. Stochastic RSI %K and %D
All aligned by timestamp
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import sys
import os

# Enable interactive matplotlib backend
import matplotlib
# Try different interactive backends
try:
    matplotlib.use('TkAgg')  # Interactive backend for zooming/panning
except ImportError:
    try:
        matplotlib.use('Qt5Agg')  # Alternative interactive backend
    except ImportError:
        try:
            matplotlib.use('MacOSX')  # macOS native backend
        except ImportError:
            print("Using default matplotlib backend (may have limited interactivity)")

# Add indicator calculator to path
sys.path.append('./indicator-calculator')
from indicator_calculator import IndicatorCalculator

# Optional: Try to import plotly for web-based interactive charts
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare options data with indicators"""
    print(f"Loading data from: {file_path}")
    
    # Load the data
    data = pd.read_csv(file_path)
    print(f"Loaded {len(data)} rows")
    
    # Convert timestamp to datetime
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    else:
        print("Error: 'timestamp' column not found")
        return None
    
    # Sort by timestamp to ensure proper order
    data = data.sort_values('timestamp').reset_index(drop=True)
    
    return data

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate all required indicators"""
    print("Calculating indicators...")
    
    calc = IndicatorCalculator()
    
    # Define indicator periods (matching your backtester config)
    indicator_periods = {
        'ema': 8,
        'vwma': 13,
        'stoch_rsi_period': 14,
        'stoch_rsi_k': 3,
        'stoch_rsi_d': 3,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'roc': 11,
        'roc_of_roc': 18
    }
    
    # Calculate all indicators using the updated platform-behavior functions
    result_data = calc.calculate_all_indicators(data, indicator_periods, 'last_price')
    
    print("Indicators calculated successfully")
    return result_data

def create_comprehensive_plot(data: pd.DataFrame, symbol: str = "Options"):
    """Create the comprehensive multi-panel plot"""
    
    # Calculate optimal width based on data points for full utilization
    data_points = len(data)
    # Use wider figure to take full advantage of screen width
    # Make height scrollable by increasing it proportionally
    fig_width = max(20, data_points / 150)  # Wider figure, scales with data
    fig_height = 16  # Taller for better visibility and scrollability
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(fig_width, fig_height), sharex=True)
    fig.suptitle(f'{symbol} - Comprehensive Technical Analysis', fontsize=18, fontweight='bold')
    
    # Prepare time axis
    time_axis = data['timestamp']
    
    # Panel 1: Price, EMA, and VWMA
    ax1 = axes[0]
    ax1.plot(time_axis, data['last_price'], label='Last Price', color='black', linewidth=1.0)
    
    if 'ema' in data.columns:
        ax1.plot(time_axis, data['ema'], label='EMA(8)', color='blue', linewidth=1.0, alpha=0.9)
    
    if 'vwma' in data.columns:
        ax1.plot(time_axis, data['vwma'], label='VWMA(13)', color='red', linewidth=1.0, alpha=0.9)
    
    ax1.set_ylabel('Price', fontweight='bold', fontsize=11)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Price Action with Moving Averages', fontsize=12, fontweight='bold')
    
    # Panel 2: ROC and ROC of ROC (overlapping)
    ax2 = axes[1]
    
    if 'roc' in data.columns:
        ax2.plot(time_axis, data['roc'], label='ROC(11)', color='green', linewidth=1.0)
    
    if 'roc_of_roc' in data.columns:
        # Filter out infinite values for better visualization
        roc_of_roc_filtered = data['roc_of_roc'].replace([float('inf'), float('-inf')], float('nan'))
        ax2.plot(time_axis, roc_of_roc_filtered, label='ROC of ROC(18)', color='orange', linewidth=1.0)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('ROC (%)', fontweight='bold', fontsize=11)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Rate of Change Indicators', fontsize=12, fontweight='bold')
    
    # Panel 3: MACD line, signal line, and histogram
    ax3 = axes[2]
    
    if 'macd_line' in data.columns and 'macd_signal' in data.columns:
        ax3.plot(time_axis, data['macd_line'], label='MACD Line', color='blue', linewidth=1.0)
        ax3.plot(time_axis, data['macd_signal'], label='Signal Line', color='red', linewidth=1.0)
        
        # MACD Histogram (MACD - Signal) - use thinner bars for wide charts
        histogram = data['macd_line'] - data['macd_signal']
        colors = ['green' if x >= 0 else 'red' for x in histogram]
        # Calculate appropriate bar width based on data density
        bar_width = (time_axis.max() - time_axis.min()).total_seconds() / len(time_axis) / 86400  # Convert to days
        ax3.bar(time_axis, histogram, label='Histogram', alpha=0.4, color=colors, width=bar_width)
    
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('MACD', fontweight='bold', fontsize=11)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('MACD (12,26,9)', fontsize=12, fontweight='bold')
    
    # Panel 4: Stochastic RSI %K and %D
    ax4 = axes[3]
    
    if 'stoch_rsi_k' in data.columns:
        ax4.plot(time_axis, data['stoch_rsi_k'], label='Stoch RSI %K', color='purple', linewidth=1.0)
    
    if 'stoch_rsi_d' in data.columns:
        ax4.plot(time_axis, data['stoch_rsi_d'], label='Stoch RSI %D', color='brown', linewidth=1.0)
    
    # Add overbought/oversold lines
    ax4.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Overbought (80)')
    ax4.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Oversold (20)')
    ax4.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    
    ax4.set_ylabel('Stoch RSI', fontweight='bold', fontsize=11)
    ax4.set_ylim(0, 100)
    ax4.legend(loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Stochastic RSI (14,3,3)', fontsize=12, fontweight='bold')
    
    # Format x-axis (time) with better spacing for wide charts
    # Adjust time formatting based on data density
    time_span = (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600  # hours
    
    if time_span <= 2:  # Less than 2 hours of data
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    elif time_span <= 8:  # Less than 8 hours
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax4.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    else:  # More than 8 hours
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax4.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, fontsize=10)
    ax4.set_xlabel('Time', fontweight='bold', fontsize=12)
    
    # Adjust layout with more space for wide charts
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, hspace=0.25)
    
    return fig

def create_interactive_plotly_chart(data: pd.DataFrame, symbol: str = "Options"):
    """Create an interactive web-based chart using Plotly"""
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Using matplotlib instead.")
        return None
    
    print("Creating interactive Plotly chart...")
    
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price Action with Moving Averages', 
                       'Rate of Change Indicators',
                       'MACD (12,26,9)',
                       'Stochastic RSI (14,3,3)'),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )
    
    # Panel 1: Price, EMA, and VWMA
    fig.add_trace(go.Scatter(x=data['timestamp'], y=data['last_price'], 
                            name='Last Price', line=dict(color='black', width=2)),
                  row=1, col=1)
    
    if 'ema' in data.columns:
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['ema'], 
                                name='EMA(8)', line=dict(color='blue', width=1.5)),
                      row=1, col=1)
    
    if 'vwma' in data.columns:
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['vwma'], 
                                name='VWMA(13)', line=dict(color='red', width=1.5)),
                      row=1, col=1)
    
    # Panel 2: ROC and ROC of ROC
    if 'roc' in data.columns:
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['roc'], 
                                name='ROC(11)', line=dict(color='green', width=1.5)),
                      row=2, col=1)
    
    if 'roc_of_roc' in data.columns:
        roc_of_roc_filtered = data['roc_of_roc'].replace([float('inf'), float('-inf')], float('nan'))
        fig.add_trace(go.Scatter(x=data['timestamp'], y=roc_of_roc_filtered, 
                                name='ROC of ROC(18)', line=dict(color='orange', width=1.5)),
                      row=2, col=1)
    
    # Add zero line for ROC
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Panel 3: MACD
    if 'macd_line' in data.columns and 'macd_signal' in data.columns:
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['macd_line'], 
                                name='MACD Line', line=dict(color='blue', width=1.5)),
                      row=3, col=1)
        
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['macd_signal'], 
                                name='Signal Line', line=dict(color='red', width=1.5)),
                      row=3, col=1)
        
        # MACD Histogram
        histogram = data['macd_line'] - data['macd_signal']
        colors = ['green' if x >= 0 else 'red' for x in histogram]
        fig.add_trace(go.Bar(x=data['timestamp'], y=histogram, 
                            name='Histogram', marker_color=colors, opacity=0.4),
                      row=3, col=1)
    
    # Add zero line for MACD
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)
    
    # Panel 4: Stochastic RSI
    if 'stoch_rsi_k' in data.columns:
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['stoch_rsi_k'], 
                                name='Stoch RSI %K', line=dict(color='purple', width=1.5)),
                      row=4, col=1)
    
    if 'stoch_rsi_d' in data.columns:
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['stoch_rsi_d'], 
                                name='Stoch RSI %D', line=dict(color='brown', width=1.5)),
                      row=4, col=1)
    
    # Add overbought/oversold lines
    fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)
    fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.3, row=4, col=1)
    
    # Update layout for interactivity
    fig.update_layout(
        title=f'{symbol} - Interactive Technical Analysis',
        height=900,
        showlegend=True,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ROC (%)", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Stoch RSI", row=4, col=1, range=[0, 100])
    
    # Update x-axis label
    fig.update_xaxes(title_text="Time", row=4, col=1)
    
    return fig

def main():
    """Main function to create the comprehensive plot"""
    
    # You can change this to any options file
    file_path = 'data/options/SPY250806C00631000.csv'
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        print("Available files in data/options/:")
        if os.path.exists('data/options/'):
            for f in os.listdir('data/options/'):
                if f.endswith('.csv'):
                    print(f"  - {f}")
        return
    
    # Load and prepare data
    data = load_and_prepare_data(file_path)
    if data is None:
        return
    
    # Calculate indicators
    data_with_indicators = calculate_indicators(data)
    
    # Extract symbol from filename
    symbol = os.path.basename(file_path).replace('.csv', '')
    
    # Ask user which type of chart they want
    print("\nChart options:")
    print("1. Interactive matplotlib chart (zoomable/pannable)")
    print("2. Interactive web-based chart (Plotly)")
    print("3. Both")
    
    try:
        choice = input("Choose chart type (1-3, or press Enter for option 1): ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"
    
    if choice in ["1", "3"]:
        # Create matplotlib chart with interactive features
        print("Creating interactive matplotlib chart...")
        fig = create_comprehensive_plot(data_with_indicators, symbol)
        
        # Enable interactive navigation toolbar
        plt.rcParams['toolbar'] = 'toolmanager'
        
        # Save the plot
        output_file = f'comprehensive_indicators_plot_{symbol}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none',
                    format='png', metadata={'Title': f'{symbol} Technical Analysis'})
        print(f"Plot saved as: {output_file}")
        print(f"Chart dimensions: {fig.get_size_inches()[0]:.1f} x {fig.get_size_inches()[1]:.1f} inches")
        
        print("\n=== Interactive Controls ===")
        print("🔍 Zoom: Use mouse wheel or zoom tool")
        print("👆 Pan: Click and drag")
        print("🏠 Home: Reset view")
        print("↩️  Back/Forward: Navigate zoom history")
        print("💾 Save: Save current view")
        
        # Show the interactive plot
        plt.show()
    
    if choice in ["2", "3"]:
        # Create interactive Plotly chart
        if PLOTLY_AVAILABLE:
            plotly_fig = create_interactive_plotly_chart(data_with_indicators, symbol)
            
            if plotly_fig:
                # Save as HTML
                html_file = f'interactive_chart_{symbol}.html'
                plotly_fig.write_html(html_file)
                print(f"Interactive chart saved as: {html_file}")
                
                print("\n=== Interactive Web Chart Features ===")
                print("🔍 Zoom: Select area to zoom")
                print("👆 Pan: Click and drag")
                print("🏠 Reset: Double-click")
                print("📱 Responsive: Works on mobile")
                print("💾 Download: Use toolbar to save")
                print("👁️  Toggle: Click legend to show/hide series")
                
                # Show in browser
                plotly_fig.show()
        else:
            print("Plotly not available. Please install with: pip install plotly")
    
    # Print some statistics
    print("\n=== Indicator Statistics ===")
    indicators = ['last_price', 'ema', 'vwma', 'macd_line', 'macd_signal', 'roc', 'roc_of_roc', 'stoch_rsi_k', 'stoch_rsi_d']
    
    for indicator in indicators:
        if indicator in data_with_indicators.columns:
            valid_data = data_with_indicators[indicator].dropna()
            if len(valid_data) > 0:
                print(f"{indicator:15s}: {len(valid_data):4d} valid values, "
                      f"starts at row {data_with_indicators[indicator].first_valid_index() + 1:2d}, "
                      f"range: {valid_data.min():.4f} to {valid_data.max():.4f}")

def plot_multiple_files():
    """Function to plot multiple options files for comparison"""
    
    options_dir = 'data/options/'
    if not os.path.exists(options_dir):
        print(f"Error: Directory {options_dir} not found")
        return
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(options_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in options directory")
        return
    
    print("Available options files:")
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {file}")
    
    # Let user choose a file
    try:
        choice = int(input(f"Choose a file (1-{len(csv_files)}): ")) - 1
        if 0 <= choice < len(csv_files):
            file_path = os.path.join(options_dir, csv_files[choice])
            
            # Update the file path in main function
            data = load_and_prepare_data(file_path)
            if data is None:
                return
            
            data_with_indicators = calculate_indicators(data)
            symbol = os.path.basename(file_path).replace('.csv', '')
            
            print("Creating comprehensive plot...")
            fig = create_comprehensive_plot(data_with_indicators, symbol)
            
            output_file = f'comprehensive_indicators_plot_{symbol}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {output_file}")
            plt.show()
            
        else:
            print("Invalid choice")
    except ValueError:
        print("Invalid input")

if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment the line below if you want to choose from multiple files interactively
    # plot_multiple_files()