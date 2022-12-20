from torch import nn

class LSTM(nn.Module):
    def __init__(self, device, input_size=9, hidden_size=32, output_size=6, num_layers=2):
        super(self.__class__, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True).to(device)
        self.classification = nn.Sequential(nn.Linear(128 * 32, output_size), 
                                nn.ReLU(),
                                nn.Softmax(-1)).to(device)
        
    def forward(self, x):
        h_seq, _ = self.lstm(x)
        output = self.classification(h_seq.reshape(x.shape[0], -1))
        return output
