#!/usr/bin/env python3
"""
Plot options data from CSV file with price vs time
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def plot_options_data(csv_file_path):
    """
    Plot options data with price on y-axis and time on x-axis
    
    Args:
        csv_file_path (str): Path to the CSV file containing options data
    """
    
    # Read the CSV file
    print(f"Reading data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp to ensure proper plotting
    df = df.sort_values('datetime')
    
    # Get the option symbol for the title
    option_symbol = df['description'].iloc[0] if 'description' in df.columns else "SPY Option"
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Price data
    ax1.plot(df['datetime'], df['last_price'], label='Last Price', linewidth=2, color='blue')
    ax1.plot(df['datetime'], df['bid_price'], label='Bid Price', linewidth=1, alpha=0.7, color='red')
    ax1.plot(df['datetime'], df['ask_price'], label='Ask Price', linewidth=1, alpha=0.7, color='green')
    
    # Fill between bid and ask to show spread
    ax1.fill_between(df['datetime'], df['bid_price'], df['ask_price'], 
                     alpha=0.2, color='gray', label='Bid-Ask Spread')
    
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title(f'Options Price Data: {option_symbol}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add price statistics as text
    price_stats = f"""
    Last Price: ${df['last_price'].iloc[-1]:.2f}
    Daily High: ${df['high_price'].max():.2f}
    Daily Low: ${df['low_price'].min():.2f}
    Avg Spread: ${(df['ask_price'] - df['bid_price']).mean():.3f}
    """
    ax1.text(0.02, 0.98, price_stats, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Volume and Open Interest
    ax2_twin = ax2.twinx()
    
    # Volume bars
    ax2.bar(df['datetime'], df['volume'], alpha=0.6, color='purple', label='Volume', width=0.0001)
    ax2.set_ylabel('Volume', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # Open Interest line
    ax2_twin.plot(df['datetime'], df['open_interest'], color='orange', linewidth=2, label='Open Interest')
    ax2_twin.set_ylabel('Open Interest', fontsize=12, color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    # Format x-axis
    ax2.set_xlabel('Time', fontsize=12)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Add legends
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and show
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"options_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    
    # Show the plot
    plt.show()
    
    # Print some basic statistics
    print("\n=== OPTIONS DATA SUMMARY ===")
    print(f"Option: {option_symbol}")
    print(f"Data Period: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Total Data Points: {len(df)}")
    print(f"Price Range: ${df['last_price'].min():.2f} - ${df['last_price'].max():.2f}")
    print(f"Total Volume: {df['volume'].sum():,}")
    print(f"Current Open Interest: {df['open_interest'].iloc[-1]:,}")
    print(f"Current Underlying Price: ${df['underlying_price'].iloc[-1]:.2f}")
    print(f"Days to Expiration: {df['days_to_expiration'].iloc[-1]}")
    print(f"Strike Price: ${df['strike_price'].iloc[0]:.2f}")
    print(f"Current Delta: {df['delta'].iloc[-1]:.3f}")
    print(f"Current Gamma: {df['gamma'].iloc[-1]:.3f}")
    print(f"Current Theta: {df['theta'].iloc[-1]:.3f}")
    print(f"Current Vega: {df['vega'].iloc[-1]:.3f}")

def main():
    """Main function to run the plotting script"""
    csv_file_path = "/Users/elkingarcia/Documents/python/market-data-backtester/data/options/SPY250728P00635000.csv"
    
    try:
        plot_options_data(csv_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"Error occurred while plotting: {e}")
        print("Please check the data format and try again.")

if __name__ == "__main__":
    main()