import pandas as pd

def clean_options_signal_combo_results():
    """
    Open options signal combo results
    drop trades column if exists
    drop rows where combo_index is empty
    filter out rows with total_trades < 5 and total trades > 12   
    filter out rows with avg_profit < 0
    sort by total_profit
    save to csv
    """
    # Open options signal combo results
    df = pd.read_csv('options_signal_combo_results_latest.csv')

    # drop trades column if exists
    if 'trades' in df.columns:
        df = df.drop(columns=['trades'])

    # drop rows where combo_index is empty
    df = df[df['combo_index'].notna()]

    # filter out rows with total_trades < 5 and total trades > 12   
    df = df[(df['total_trades'] >= 5) & (df['total_trades'] <= 12)]

    # filter out rows with avg_profit < 0
    df = df[df['avg_profit'] >= 0]

    # sort by total_profit
    df = df.sort_values(by='total_profit', ascending=False)

    # save to csv
    df.to_csv('options_signal_combo_results_latest_clean.csv', index=False)

    print(f"Filtered data saved. Final dataset has {len(df)} rows.")

if __name__ == "__main__":
    clean_options_signal_combo_results()