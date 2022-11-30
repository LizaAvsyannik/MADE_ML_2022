import torch
import torch.nn.functional as F
from torch import nn


class TinyNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            # Your network structure comes here
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(),
            nn.BatchNorm2d(num_features=64),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, inp):       
        return self.model(inp)


class OverfittingNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            # Your network structure comes here
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, inp):       
        return self.model(inp)


class FixedNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            # Your network structure comes here
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(),
            nn.BatchNorm2d(num_features=128),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, inp):       
        return self.model(inp)


class CompletelyFixedNeuralNetwork(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            # Your network structure comes here
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(num_features=128),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, inp):       
        return self.model(inp)


class RNN(nn.Module):
    def __init__(self, device, input_size=41, hidden_size=64):
        super(self.__class__, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True).to(device)
        self.hid_to_logits = nn.Linear(hidden_size, input_size).to(device)
        
    def forward(self, x, h_state):
        emb = F.one_hot(x, self.input_size).float()
        h_seq, h_state = self.rnn(emb, h_state)
        next_logits = self.hid_to_logits(h_seq)
        return next_logits, h_state
    
    def initial_state(self, batch_size=1):
        if batch_size != 1:
            return torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        else:
            return torch.zeros(batch_size, self.hidden_size).to(self.device)


class LSTM(nn.Module):
    def __init__(self, device, input_size=41, hidden_size=64, num_layers=2):
        super(self.__class__, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True).to(device)
        self.hid_to_logits = nn.Linear(hidden_size, input_size).to(device)
        
    def forward(self, x, state):
        emb = F.one_hot(x, self.input_size).float()
        h_seq, state = self.lstm(emb, state)
        next_logits = self.hid_to_logits(h_seq)
        return next_logits, state
    
    def initial_state(self, batch_size=1):
        if batch_size != 1:
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device), \
                    torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))
        else:
            return (torch.zeros(self.num_layers, self.hidden_size).to(self.device), \
                    torch.zeros(self.num_layers, self.hidden_size).to(self.device))
