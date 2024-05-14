# 420 * 2 network input (420 datapoints for one adj_price, one for volume)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import torch 
import torch.nn as nn  
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from lstm import LSTM

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def read_data(filename):
    data = pd.read_csv(filename)
    df = data[["Datetime", 'Adj Close', 'Volume']]

    plt.figure(figsize=(14,7))
    plt.plot(df['Adj Close'], label='Adj Close')
    # plt.plot(df['Volume'], label='Volume')
    plt.title("AAPL Adj Close Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    # plt.show()
    return df

def prepare_dataframe_for_lstm(df, n_steps):
   # "t-1, t-2, t-3..."
    df = dc(df)
    df.set_index('Datetime', inplace=True)
    for i in range(1, n_steps+1):
        df[f'Adj Close(t-{i})'] = df['Adj Close'].shift(i)
        df[f'Volume(t-{i})'] = df['Volume'].shift(i)
    df.dropna(inplace=True)
    
    return df

def scale_dataframe(np_array):
    # scale the data to be between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    numpy_df_scaled = scaler.fit_transform(np_array)
    return numpy_df_scaled

def torch_data_prepare(scaled_np_array, lookback):
    x = scaled_numpy[:, 1:]
    y = scaled_numpy[:, :1]
    print(x.shape, y.shape)

    split_index = int(0.8 * len(scaled_numpy))

    x_train = x[:split_index]
    x_test = x[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    x_train = x_train.reshape((-1, lookback, 1))
    x_test = x_test.reshape((-1, lookback, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()

    train_dataset = TimeSeriesDataset(x_train, y_train)
    test_dataset = TimeSeriesDataset(x_test, y_test)

    return train_dataset, test_dataset



if __name__ == '__main__':
    lookback = 7
    device = torch.device("cpu")
    df = read_data("combined_data.csv")
    df = prepare_dataframe_for_lstm(df, lookback)
    print("first two columns")
    df.to_csv("lstm_data.csv")
    numpy_df = df.to_numpy()
    scaled_numpy = scale_dataframe(numpy_df)
    train_dataset, test_dataset = torch_data_prepare(scaled_numpy, lookback)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM(1, 4, 1, 0.001, 10)
    model.to(device)
    
    for epoch in range(10):
        model.train_one_epoch(epoch, train_loader)
        model.validate_one_epoch(test_loader)


