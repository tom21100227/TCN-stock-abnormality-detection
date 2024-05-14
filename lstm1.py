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
    



    
if __name__ == "__main__":
    dq_np = prepare_data()
    x_train = torch.from_numpy(y[3\:, :-1])
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