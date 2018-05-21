class Discriminator(object):
    def __init__(self, embedding_size):
        self.in_channels = 1
        self.out_channels = 3  # to check
        self.kernel_sizes = [1, 2, 3]  # to check
        self.embedding_size = embedding_size
        self.hidden_size = 5
        self.dropout = 0.5
        self.learning_rate = 0.0001


class Autoencoder(object):
    def __init__(self, embedding_size, dim_y, dim_z):
        self.input_size = embedding_size
        self.hidden_size = dim_y + dim_z
        self.num_layers = 1
        self.dropout = 0.5
        self.learning_rate = 0.0001
        self.betas = (0.5, 0.999)


class Params(object):
    embedding_size = 200
    dim_y = 200
    dim_z = 500
    batch_size = 128
    temperature = 0.001
    lambda_GAN = 1
    autoencoder = Autoencoder(embedding_size, dim_y, dim_z)
    discriminator = Discriminator(embedding_size)
