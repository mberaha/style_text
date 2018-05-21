class Discriminator(object):
    def __init__(
        self, embedding_size, dropout):
        self.in_channels = 1
        self.out_channels = 3  # to check
        self.kernel_sizes = [1, 2, 3]  # to check
        self.embedding_size = embedding_size
        self.hidden_size = 5
        self.dropout = dropout


class GRU(object):
    def __init__(self, embedding_size, dim_y, dim_z):
        self.input_size = embedding_size
        self.hidden_size = dim_y + dim_z
        self.num_layers = 3


class Params(object):
    embedding_size = 200
    dim_y = 200
    dim_z = 500
    batch_size = 128
    dropout = 0.5
    learning_rate = 0.0001
    temperature = 0.001
    lambda_GAN = 1
    gru = GRU(embedding_size, dim_y, dim_z)
    discriminator = Discriminator(embedding_size, dropout)
