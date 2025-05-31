import yfinance as yf
import pandas as pd
import os 
from Tickers import tickers


# This script downloads daily stock prices for a list of tickers from Yahoo Finance,
# processes the data to calculate monthly prices and returns, and saves the results to CSV files. 

start_date = "2021-03-01"
end_date = "2025-05-01"

output_folder = "StockReturns"
os.makedirs(output_folder, exist_ok=True)
print(f"Output will be saved to: {os.path.abspath(output_folder)}")
prices_csv_path = os.path.join(output_folder, "omxs30_monthly_prices.csv")
returns_csv_path = os.path.join(output_folder, "omxs30_monthly_returns.csv")

print(f"Downloading daily data for {len(tickers)} tickers...")
daily_data = pd.DataFrame()
try:
    data_downloaded = yf.download(tickers, start=start_date, end=end_date)

    if data_downloaded.empty:
        print("Warning: yf.download returned an empty DataFrame.")
    elif 'Adj Close' not in data_downloaded.columns:
         print("Warning: 'Adj Close' column not found in downloaded data. Check data format.")
         print("Available columns:", data_downloaded.columns)
         if 'Close' in data_downloaded.columns:
             print("Using 'Close' price instead of 'Adj Close'.")
             daily_data = data_downloaded['Close']
         else:
              print("Error: Neither 'Adj Close' nor 'Close' found.")
    else:
        daily_data = data_downloaded['Adj Close']
        print("Download attempt finished.")

    if isinstance(daily_data.columns, pd.MultiIndex):
        downloaded_tickers = daily_data.columns.get_level_values(0).unique().tolist()
    else:
        downloaded_tickers = daily_data.columns.tolist()

    original_tickers_set = set(tickers)
    downloaded_tickers_set = set(downloaded_tickers) | set([t.replace('.', '-') for t in downloaded_tickers])

    missing_tickers = list(original_tickers_set - downloaded_tickers_set)

    if missing_tickers:
        print(f"\nWarning: Could not confirm download for: {missing_tickers}")

    if not daily_data.empty:
        daily_data.dropna(axis=1, how='all', inplace=True)
        if daily_data.empty:
             print("Warning: DataFrame became empty after dropping all-NaN columns.")


except Exception as e:
    print(f"An error occurred during download: {e}")

if not daily_data.empty:
    print("\nProcessing downloaded data...")
    monthly_prices = daily_data.resample('M').last()
    monthly_returns = monthly_prices.pct_change()
    monthly_returns = monthly_returns.dropna(how='all')
    try:
        monthly_prices.to_csv(prices_csv_path)
        monthly_returns.to_csv(returns_csv_path)
    except Exception as e:
        print(f"\nError saving files to CSV: {e}")

else:
    print("\nThe dataframe is empty.")