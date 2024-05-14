import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(in_channels, 64, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.pointwise_conv = nn.Conv1d(64, 16, 1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.relu1(out)
        out = self.pointwise_conv(out)
        out = self.relu2(out)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2):
        super(TemporalConvNet, self).__init__()
        self.num_levels = len(num_channels)
        self.temporal_blocks = nn.ModuleList()
        self.channel_reduction = nn.ModuleList()

        for i in range(self.num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else 16
            self.temporal_blocks.append(
                TemporalBlock(in_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size)
            )
            self.channel_reduction.append(nn.Conv1d(16, 16, 1))

    def forward(self, x):
        outputs = []
        for i in range(self.num_levels):
            x = self.temporal_blocks[i](x)
            x = self.channel_reduction[i](x)
            outputs.append(x)

        concatenated = torch.cat(outputs, dim=1)
        return concatenated

class Encoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2):
        super(Encoder, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size)
        
    def forward(self, x):
        return self.tcn(x)

class Decoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2):
        super(Decoder, self).__init__()
        reversed_channels = num_channels[::-1]
        self.tcn = TemporalConvNet(input_size, reversed_channels, kernel_size)
        
    def forward(self, x):
        return self.tcn(x)

class TCNAutoencoder(nn.Module):
    def __init__(self, input_dim=(420, 2)):
        super(TCNAutoencoder, self).__init__()
        self.encoder = Encoder(input_size=input_dim[1], num_channels=[16, 16, 16, 16, 16, 16, 16], kernel_size=2)
        self.decoder = Decoder(input_size=16 * 7, num_channels=[16, 16, 16, 16, 16, 16, 16], kernel_size=2)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded