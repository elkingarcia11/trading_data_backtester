#!/usr/bin/env python3
"""
Focused price action chart with moving averages
Shows only price, EMA, and VWMA in a large, clear format
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
    """Calculate EMA and VWMA indicators"""
    print("Calculating indicators...")
    
    calc = IndicatorCalculator()
    
    # Define indicator periods - just EMA and VWMA for price action
    indicator_periods = {
        'ema': 8,
        'vwma': 13
    }
    
    # Calculate indicators
    result_data = calc.calculate_all_indicators(data, indicator_periods, 'last_price')
    
    print("Indicators calculated successfully")
    return result_data

def create_price_action_chart(data: pd.DataFrame, symbol: str = "Options"):
    """Create a large, focused price action chart"""
    
    # Calculate optimal width based on data points for full utilization
    data_points = len(data)
    # Use much wider figure for price action focus
    fig_width = max(25, data_points / 100)  # Even wider for price focus
    fig_height = 10  # Good height for price action
    
    # Create single subplot for price action
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    fig.suptitle(f'{symbol} - Price Action with Moving Averages', fontsize=20, fontweight='bold')
    
    # Prepare time axis
    time_axis = data['timestamp']
    
    # Plot price and moving averages with ultra-thin lines
    ax.plot(time_axis, data['last_price'], label='Last Price', color='black', linewidth=0.5, zorder=3)
    
    if 'ema' in data.columns:
        ax.plot(time_axis, data['ema'], label='EMA', color='#1f77b4', linewidth=0.5, alpha=0.9, zorder=2)
    
    if 'vwma' in data.columns:
        ax.plot(time_axis, data['vwma'], label='VWMA', color='#d62728', linewidth=0.5, alpha=0.9, zorder=1)
    
    # Add vertical bars where EMA > VWMA
    if 'ema' in data.columns and 'vwma' in data.columns:
        # Find points where EMA > VWMA
        ema_above_vwma = data['ema'] > data['vwma']
        
        # Get the y-axis limits for full-height bars
        y_min = data['last_price'].min()
        y_max = data['last_price'].max()
        y_range = y_max - y_min
        y_bottom = y_min - (y_range * 0.05)  # Extend slightly below
        y_top = y_max + (y_range * 0.05)     # Extend slightly above
        
        # Add vertical bars where condition is true
        ema_above_count = 0
        for i, (timestamp, condition) in enumerate(zip(time_axis, ema_above_vwma)):
            if condition and not pd.isna(condition):
                # Only add label to the first occurrence for legend
                label = 'EMA > VWMA' if ema_above_count == 0 else None
                ax.axvline(x=timestamp, color='green', alpha=0.15, linewidth=0.8, zorder=0, label=label)
                ema_above_count += 1
    
    # Enhanced styling for better visibility
    ax.set_ylabel('Price ($)', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Format x-axis with better spacing for wide charts
    time_span = (data['timestamp'].max() - data['timestamp'].min()).total_seconds() / 3600  # hours
    
    if time_span <= 2:  # Less than 2 hours of data
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    elif time_span <= 8:  # Less than 8 hours
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    else:  # More than 8 hours
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=11)
    ax.set_xlabel('Time', fontweight='bold', fontsize=14)
    
    # Improve layout with more space
    plt.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.12)
    
    # Add some padding to y-axis for better visibility
    price_range = data['last_price'].max() - data['last_price'].min()
    padding = price_range * 0.05
    ax.set_ylim(data['last_price'].min() - padding, data['last_price'].max() + padding)
    
    return fig

def create_interactive_plotly_price_chart(data: pd.DataFrame, symbol: str = "Options"):
    """Create an interactive web-based price action chart using Plotly"""
    
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Using matplotlib instead.")
        return None
    
    print("Creating interactive Plotly price action chart...")
    
    # Create single plot
    fig = go.Figure()
    
    # Add price line with ultra-thin width
    fig.add_trace(go.Scatter(
        x=data['timestamp'], 
        y=data['last_price'], 
        name='Last Price',
        line=dict(color='black', width=0.5),
        hovertemplate='<b>Price</b>: $%{y:.4f}<br><b>Time</b>: %{x}<extra></extra>'
    ))
    
    # Add EMA
    if 'ema' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['timestamp'], 
            y=data['ema'], 
            name='EMA',
            line=dict(color='blue', width=0.5),
            hovertemplate='<b>EMA</b>: $%{y:.4f}<br><b>Time</b>: %{x}<extra></extra>'
        ))
    
    # Add VWMA
    if 'vwma' in data.columns:
        fig.add_trace(go.Scatter(
            x=data['timestamp'], 
            y=data['vwma'], 
            name='VWMA',
            line=dict(color='red', width=0.5),
            hovertemplate='<b>VWMA</b>: $%{y:.4f}<br><b>Time</b>: %{x}<extra></extra>'
        ))
    
    # Add vertical lines where EMA > VWMA
    if 'ema' in data.columns and 'vwma' in data.columns:
        ema_above_vwma = data['ema'] > data['vwma']
        
        # Add vertical lines for each point where EMA > VWMA
        for i, (timestamp, condition) in enumerate(zip(data['timestamp'], ema_above_vwma)):
            if condition and not pd.isna(condition):
                fig.add_vline(
                    x=timestamp,
                    line=dict(color='green', width=0.5),
                    opacity=0.2
                )
    
    # Update layout for better price action visibility
    fig.update_layout(
        title=f'{symbol} - Interactive Price Action with Moving Averages',
        title_font_size=20,
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        xaxis_title='Time',
        yaxis_title='Price ($)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range selector for easy navigation
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=2, label="2h", step="hour", stepmode="backward"),
                    dict(count=4, label="4h", step="hour", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=False),
            type="date"
        )
    )
    
    return fig

def main():
    """Main function to create the focused price action chart"""
    
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
    print("\nPrice Action Chart Options:")
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
        # Create matplotlib chart
        print("Creating interactive matplotlib price action chart...")
        fig = create_price_action_chart(data_with_indicators, symbol)
        
        # Save the plot
        output_file = f'price_action_chart_{symbol}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none',
                    format='png', metadata={'Title': f'{symbol} Price Action'})
        print(f"Chart saved as: {output_file}")
        print(f"Chart dimensions: {fig.get_size_inches()[0]:.1f} x {fig.get_size_inches()[1]:.1f} inches")
        
        print("\n=== Interactive Controls ===")
        print("🔍 Zoom: Use mouse wheel or zoom tool")
        print("👆 Pan: Click and drag to scroll horizontally")
        print("🏠 Home: Reset view")
        print("↩️  Back/Forward: Navigate zoom history")
        print("💾 Save: Save current view")
        
        # Show the interactive plot
        plt.show()
    
    if choice in ["2", "3"]:
        # Create interactive Plotly chart
        if PLOTLY_AVAILABLE:
            plotly_fig = create_interactive_plotly_price_chart(data_with_indicators, symbol)
            
            if plotly_fig:
                # Save as HTML
                html_file = f'interactive_price_action_{symbol}.html'
                plotly_fig.write_html(html_file)
                print(f"Interactive price action chart saved as: {html_file}")
                
                print("\n=== Interactive Web Chart Features ===")
                print("🔍 Zoom: Select area to zoom in")
                print("👆 Pan: Click and drag to scroll")
                print("🏠 Reset: Double-click to reset view")
                print("📱 Responsive: Works on mobile devices")
                print("💾 Download: Use toolbar to save as PNG")
                print("👁️  Toggle: Click legend to show/hide lines")
                print("⏰ Time Range: Use buttons for quick time selections")
                
                # Show in browser
                plotly_fig.show()
        else:
            print("Plotly not available. Please install with: pip install plotly")
    
    # Print some statistics
    print("\n=== Price Action Statistics ===")
    indicators = ['last_price', 'ema', 'vwma']
    
    for indicator in indicators:
        if indicator in data_with_indicators.columns:
            valid_data = data_with_indicators[indicator].dropna()
            if len(valid_data) > 0:
                print(f"{indicator:15s}: {len(valid_data):4d} valid values, "
                      f"starts at row {data_with_indicators[indicator].first_valid_index() + 1:2d}, "
                      f"range: ${valid_data.min():.4f} to ${valid_data.max():.4f}")

if __name__ == "__main__":
    main()