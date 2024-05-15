#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

from tcnAutoencoder import TCNAutoencoder


# In[2]:


# if in jupyter notebook, don't run this cell

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ticker", default="*", type=str, help="Ticker value")
parser.add_argument("--alias", type=str, help="Alias value")
parser.add_argument("--epochs", type=int, help="Number of epochs to train the model")
args = parser.parse_args()

# Access the values of the arguments
ticker = args.ticker
alias = args.alias
epochs = args.epochs

# Print the values for testing
print("Ticker:", ticker)
print("Alias:", alias)


# In[3]:


# %load load_data.py
import pandas as pd
import glob, os, re


# Read the data only once.  It's big!
csv_files = glob.glob(os.path.join(".", "data", "hft_data", ticker, "*_message_*.csv"))
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


# In[4]:


def prep_data(df) -> pd.DataFrame:
    df = df[['Price', 'Size']]
    df.head()

    # sample every 100ms, and the size would be the sum of the size in that 100ms. 
    # Price would be the average price in that 100ms.
    df = df.resample('100ms').agg({'Price': 'mean', 'Size': 'sum'})

    # Check for NaN values

    # forwardfill all NaN values in the data
    df = df.ffill()

    # normalize the data with mean and std
    mean = df['Price'].mean()
    std = df['Price'].std()
    df['Price'] = (df['Price'] - mean) / std

    mean = df['Size'].mean()
    std = df['Size'].std()
    df['Size'] = (df['Size'] - mean) / std

    print("original shape: ", df.shape)

    df = df.values
    # Create a tensor for every 30 minutes of data
    tensors = []
    for i in range(0, len(df), 6000):
        if i + 18000 <= len(df):
            # flip the first and second dimension, so that the shape is (batch_size, channel, sequence_length)
            tensors.append(torch.tensor(df[i:i+18000], dtype = torch.float32).unsqueeze(0))
        # if it's less than 30 minutes, then we just ignore it. 
            

    return tensors

    # Create the final torch tensor, every 1 hour is a sequence

tensors_list = []
for df in df_list[1:]: 
    tensors_list.extend(prep_data(df))


# In[5]:


# filp the first and second dimension, so that the shape is (batch_size, channel, sequence_length)
tensors_list = [tensor.permute(0, 2, 1) for tensor in tensors_list]

print(f"{len(tensors_list) = }")

dims = [tensor.shape[1] for tensor in tensors_list]

# check if all dims are the same
print(all(x == dims[0] for x in dims))


# In[6]:


# Start the training process with tensors.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device {device}")

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)

model = TCNAutoencoder(input_dim=(2, 18000)).to(device)
model.apply(initialize_weights)

criterion = LogCoshLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

import tqdm

# Randomly sample 0.2 of the data from the batch for testing, excluding them for traning.
tensors = tensors_list
tensors_train = tensors[:int(len(tensors) * 0.8)]
tensors_test = tensors[int(len(tensors) * 0.8):]


# In[7]:


def check_gradients(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    print(f'Total Gradient Norm: {total_norm}')


# In[ ]:


# Train the model

import time
losses = []
test_losses = []
for epoch in range(epochs):
    # time each epoch
    start_time = time.time()

    # shuffle the data
    np.random.shuffle(tensors_train)

    # create a mini-batch of 32
    data = torch.cat(tensors_train[:32], dim=0).to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)
    loss.backward()
    # This line is not needed as my model does not have exploding gradients
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
    losses.append(loss.item())
    print(f"Time taken (seconds): {time.time() - start_time}")
    if epoch % 40 == 0:
        # test the model
        model.eval()
        check_gradients(model)
        test_data = torch.cat(tensors_test, dim=0).to(device)
        test_output = model(test_data)
        test_loss = criterion(test_output, test_data)
        print(f"\t\tTest Loss: {test_loss.item()}")
        # save a model
        test_losses.append((epoch, test_loss.item()))
        torch.save(model.state_dict(), f"{alias}_model_epoch_{epoch}.plt")
        print(f"Model saved as {alias}_model_epoch_{epoch}.plt")
        model.train()


# final OOS test
model.eval()
check_gradients(model)
test_data = torch.cat(tensors_test, dim=0).to(device)
test_output = model(test_data)
test_loss = criterion(test_output, test_data)
print(f"\t\tTest Loss: {test_loss.item()}")
# save a model
test_losses.append((epoch, test_loss.item()))

# save a final model
torch.save(model.state_dict(), f"{alias}_model_final.plt")
print(f"Model saved as {alias}_model_final.plt")

    


# In[11]:


# plot the loss
epochs, t_losses = zip(*test_losses)
plt.plot(epochs, t_losses)
# overlay the training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Test Loss', 'Train Loss'])
# save the plot in the current directory
plt.savefig(f'{alias}loss_plot.png')
print(f"Loss plot saved as {alias}loss_plot.png")


# Run another 1000 epochs


# In[ ]:


plt.plot(losses)


# In[ ]:




