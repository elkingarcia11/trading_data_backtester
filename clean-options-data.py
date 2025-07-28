# Loop through all csv files in the data/options directory
# Delete all macd_line,ema,roc,macd_signal,vwma,macd_histogram,roc_of_roc columns
# Save back to the same file
import os
import pandas as pd

def clean_options_data(filepath):
    # Loop through all csv files in the data/options directory
    for file in os.listdir(filepath):
        # Read the file
        df = pd.read_csv(f'{filepath}/{file}')
        # Delete the columns
        df = df.drop(columns=['macd_line', 'ema', 'roc', 'macd_signal', 'vwma', 'macd_histogram', 'roc_of_roc'])
        # Save back to the same file
        df.to_csv(f'{filepath}/{file}', index=False)

clean_options_data('data/options')