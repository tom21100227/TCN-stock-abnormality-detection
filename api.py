# yfinance requires pandas 1.3.5, breaks with 1.4.0
import yfinance as yf
import pandas as pd

# # Fetch data for all valid stock symbols (symbols as of 2023-07-14).
# with open("valid_stocks.txt", "r") as f:
#     tickers = f.read().splitlines()
tickers = ["AAPL"]


start = ["2024-04-14", "2024-04-21", "2024-04-28", "2024-05-05"]
end = ["2024-04-21", "2024-04-28", "2024-05-05", "2024-05-12"]

combined_data = pd.DataFrame()

for i in range(4):
    for ticker in tickers:
        data = yf.download(ticker, start=start[i], end=end[i], group_by="Ticker", period="max", interval="1m")
        data.to_csv(f'/Users/linguoren/Documents/GitHub/final_proj/{ticker}_{i}.csv', float_format='%.2f')
        df = pd.read_csv(f'/Users/linguoren/Documents/GitHub/final_proj/{ticker}_{i}.csv')
        combined_data = pd.concat([combined_data, df])

combined_data.to_csv('/Users/linguoren/Documents/GitHub/final_proj/combined_data.csv', index=False)
