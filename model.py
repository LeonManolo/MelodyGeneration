import torch
import torch.nn as nn

class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(MusicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        # Vielleicht ein GRU = Gated Recurrent Network?
        #self.lstm = nn.GRU(input_size,hidden_size,num_layers,batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # LSTM forward pass
        out, hidden = self.lstm(x, hidden)
        # Flatten LSTM output into the output size
        out = self.fc(out[:, -1, :])  # Take the last time-step's output
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)