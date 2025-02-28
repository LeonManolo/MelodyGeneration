import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MusicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        super(MusicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Mehrschichtige LSTM mit Dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Fully connected layer zur Projektion auf den Ausgabe-Vektor
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        # Verwende den Output der letzten Zeitschritt
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialisiere die verborgenen Zustände
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)
