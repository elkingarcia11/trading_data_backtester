import os
import pandas as pd

def clean_options_data(filepath, columns_to_drop=['ema', 'vwma', 'roc', 'roc_of_roc', 'macd_line', 'macd_signal', 'stoch_rsi_k', 'stoch_rsi_d']):
    """
    Loop through all csv files in the data/options directory and delete all macd_line,ema,roc,macd_signal,vwma,macd_histogram,roc_of_roc columns
    """
    for file in os.listdir(filepath):
        if not file.endswith('.csv'):
            continue
            
        print(f"Processing {file}...")
        
        # Try different encodings to handle the Unicode error
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(f'{filepath}/{file}', encoding=encoding)
                print(f"  Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"  Failed with {encoding} encoding, trying next...")
                continue
            except Exception as e:
                print(f"  Error reading {file}: {e}")
                break
        
        if df is None:
            print(f"  Could not read {file} with any encoding, skipping...")
            continue
            
        # Check which columns exist before trying to drop them
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        
        if existing_columns:
            print(f"  Dropping columns: {existing_columns}")
            df = df.drop(columns=existing_columns)
        else:
            print(f"  No columns to drop found in {file}")
        
        # Save back to the same file with UTF-8 encoding
        try:
            df.to_csv(f'{filepath}/{file}', index=False, encoding='utf-8')
            print(f"  Successfully saved {file}")
        except Exception as e:
            print(f"  Error saving {file}: {e}")

if __name__ == "__main__":
    clean_options_data('data/options')