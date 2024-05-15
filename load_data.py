import numpy as np
import pandas as pd
import glob, os, re


# Read the data only once.  It's big!
csv_files = glob.glob(os.path.join(".", "data", "hft_data", "*", "*_message_*.csv"))
date_str = re.compile(r'_(\d{4}-\d{2}-\d{2})_')
stock_str = re.compile(r'([A-Z]+)_\d{4}-\d{2}-\d{2}_')

df_list = []
day_list = []
sym_list = []

for csv_file in sorted(csv_files):
    date = date_str.search(csv_file)
    date = date.group(1)
    day_list.append(date)

    symbol = stock_str.search(csv_file)
    symbol = symbol.group(1)
    sym_list.append(symbol)

    # Find the order book file that matches this message file.
    book_file = csv_file.replace("message", "orderbook")

    # Read the message file and index by timestamp.
    df = pd.read_csv(csv_file, names=['Time','EventType','OrderID','Size','Price','Direction'])
    df['Time'] = pd.to_datetime(date) + pd.to_timedelta(df['Time'], unit='s')

    # Read the order book file and merge it with the messages.
    names = [f"{x}{i}" for i in range(1,11) for x in ["AP","AS","BP","BS"]]
    df = df.join(pd.read_csv(book_file, names=names), how='inner')
    df = df.set_index(['Time'])

    BBID_COL = df.columns.get_loc("BP1")
    BASK_COL = df.columns.get_loc("AP1")

    print (f"Read {df.shape[0]} unique order book shapshots from {csv_file}")

    df_list.append(df)

days = len(day_list)
