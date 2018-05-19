class GeneratorParams(object):
    input_size = 100
    hidden_size = 100
    num_layers = 3


class EncoderParams(object):
    input_size = 100
    hidden_size = 100
    num_layers = 3


class Params(object):
    generator = GeneratorParams()
    encoder = EncoderParams()
    batch_size = 128
    dim_y = 200
