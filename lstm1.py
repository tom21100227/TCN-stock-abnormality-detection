import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import glob, os, re

N = 100
L = 1000
T =20

x = np.empty((N, L), np.float32)
x[:] = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
y = np.sin(x/1.0/T).astype(np.float32)

plt.figure()
plt.plot(np.arange(x.shape[1]), y[0, : ], "r", linewidth=2.0)
plt.legend()
# plt.show()

class LSTMPredictor(nn.Module):
    def __init__(self, n_hidden = 10):
        super(LSTMPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.lstm1 = nn.LSTMCell(1, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, 1)

    def forward(self, x, future = 0):
        outputs = []
        n_samples = x.size(0)

        h_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)

        for input_t in x.split(1, dim = 1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)   

        outputs = torch.cat(outputs, dim = 1)
        return outputs
    

def prepare_data():
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
        

        #prepare the data to be trained 
        dq_df = df.loc[:, ['AP1', 'AS1', 'BP1', 'BS1']]
        dq_df["AP1"] = dq_df["AP1"] / 10000
        dq_df["BP1"] = dq_df["BP1"] / 10000
        dq_df["mid_price"] = (dq_df['AP1'] + dq_df['BP1']) / 2
        # print(dq_df)
        # dq_np = dq_df.to_numpy()
    return dq_np
    

    
if __name__ == "__main__":
    dq_np = prepare_data()
    print(dq_np)
    x_train = torch.from_numpy(y[3:, :-1])
    y_train = torch.from_numpy(y[3:, 1:])
    x_test = torch.from_numpy(y[:3, :-1])
    y_test = torch.from_numpy(y[:3, 1:])
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    model = LSTMPredictor()
    loss_fxn = nn.MSELoss()
    #limited meory BFGS: whole data, need a closue function as an input
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 10

    for i in range(n_steps):
        print(f"Step {i}")
        def closure():
            optimizer.zero_grad()
            out = model(x_train)
            loss = loss_fxn(out, y_train)
            print("loss:", loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        future = 1000
        pred = model(x_test, future = future)
        loss = loss_fxn(pred[:, :-future], y_test)
        print("test loss:", loss.item())
        y = pred.detach().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Step{i+1}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    n = x_train.shape[1] #999

    def draw(y_i, color):
        plt.plot(np.arange(n), y_i[:n], color, linewidth=2.0)
        plt.plot(np.arange(n, n+future), y_i[n:], color + ":", linewidth=2.0)

    draw(y[0], "r")
    draw(y[1], "g")
    draw(y[2], "b")
    plt.show()
    plt.savefig("/Users/linguoren/Documents/GitHub/final_proj/predict_future.pdf" % i)
    plt.close()