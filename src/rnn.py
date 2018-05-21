import torch
from torch import nn


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        # call the __init__ of nn.Module and inherit its functions
        super().__init__()
        # device storing hidden states
        self.device = device
        # size of embeddings
        self.input_size = input_size
        # size of the GRU ouput = y + z
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # create the RNN cell
        self.cell = nn.GRU(
            self.input_size, self.hidden_size, self.num_layers)

    def forward(self, input, hidden):
        output, hidden = self.cell(input, hidden)
        return output, hidden
