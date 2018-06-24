class DiscriminatorParams(object):
    def __init__(self, embedding_size):
        self.in_channels = 1
        self.out_channels = 3  # default in paper=128
        self.kernel_sizes = [1, 2, 3]  # default in paper=[1, 2, 3]
        self.embedding_size = embedding_size
        self.hidden_size = 5
        self.dropout = 0.5
        self.learning_rate = 0.001
        self.betas = (0.5, 0.999)


class AutoencoderParams(object):
    def __init__(self, embedding_size, dim_y, dim_z):
        self.input_size = embedding_size
        self.hidden_size = dim_y + dim_z
        self.num_layers = 1
        self.dropout = 0.5
        self.learning_rate = 0.0001
        self.betas = (0.5, 0.999)


class Params(object):
    in_memory = True
    max_len = 20
    embedding_size = 200
    dim_y = 200
    dim_z = 500
    batch_size = 4
    epochs = 20  # 2 for debugging, then go to 10
    temperature = 0.001
    lambda_GAN = 1
    dropout = 0.5
    max_loss = 1e10
    grad_clip = 20
    max_d_loss = 1.2
    savefile = "data/yelp/model"
    autoencoder = AutoencoderParams(embedding_size, dim_y, dim_z)
    discriminator = DiscriminatorParams(embedding_size)
