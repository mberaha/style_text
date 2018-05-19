from src.rnn import Rnn


class Generator(Rnn):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(input_size, hidden_size, num_layers)
