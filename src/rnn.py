from torch import nn


class Rnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        # call the __init__ of nn.Module and inherit its functions
        super().__init__()
        # size of embeddings
        self.input_size = input_size
        # size of the GRU ouput = y + z
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # create the RNN cell
        self.batch_first = batch_first
        self.cell = nn.GRU(
            self.input_size, self.hidden_size,
            self.num_layers, batch_first=self.batch_first)
        # print("Batch first: ", self.cell.batch_first)

    def forward(self, inputs, hidden, lengths=[], pad=True):
        if pad:
            inputs = nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=self.batch_first)

        output, hidden = self.cell(inputs, hidden)

        if pad:
            output = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=self.batch_first)[0]
        return output, hidden
