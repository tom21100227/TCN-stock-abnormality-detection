import torch
from torch import nn
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, lr, num_epoc):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lr = lr
        self.num_epoc = num_epoc

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device("cpu")
        self.to(self.device)



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, lr, num_epoc):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lr = lr
        self.num_epoc = num_epoc

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
   
        return out

    def train_one_epoch(self, epoch, train_loader):
        self.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0
        
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            
            output = self(x_batch)
            loss = self.loss(output, y_batch)
            running_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()

    def validate_one_epoch(self, test_loader):
        self.train(False)
        running_loss = 0.0
        
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            
            with torch.no_grad():
                output = self(x_batch)
                loss = self.loss(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)
        
        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print()
    
    def train_one_epoch(self, epoch, train_loader):
        self.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0
        
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            
            output = self(x_batch)
            loss = self.loss(output, y_batch)
            running_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()

    def validate_one_epoch(self, test_loader):
        self.train(False)
        running_loss = 0.0
        
        for batch_index, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            
            with torch.no_grad():
                output = self(x_batch)
                loss = self.loss(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)
        
        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print()