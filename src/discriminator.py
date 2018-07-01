import torch
import torch.nn as nn
import torch.nn.functional as F


class Cnn(nn.Module):
    """
        GAN discriminator is a TextCNN
    """

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 hidden_size, dropout):
        """
        Args:
        in_channels -- the input feature maps. Should be only one for text.
        out_channels -- the output feature maps a.k.a response maps
                        = number of filters/kernels
        kernel_sizes -- the lengths of the filters. Should be the number of
                        generators' hidden states sweeped at a time
                        by the different filters.
        hidden_size -- size of the hidden states of the generator
        hidden_units = is the number of hidden units for the Linear layer
        """
        super().__init__()

        self.dropoutLayer = nn.Dropout(p=dropout)
        # build parallel CNNs with different kernel sizes
        self.convs = nn.ModuleList([])  # CHECK CONVS ON ORIGINAL PAPER
        for ks in kernel_sizes:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(ks, hidden_size),
                stride=1)
            self.convs.append(conv)
        self.linear = nn.Linear(out_channels*len(kernel_sizes), 1)

    def forward(self, x):
        """
        Args:
        x -- (batch_size, in_channels=1, seq_len, hidden_size)
            = (1, 1, seq_length (max_length for professor), hidden_size)
        """
        x = [
            F.leaky_relu(conv(x), negative_slope=0.01).squeeze(3)
            for conv in self.convs]
        x = [
            F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.dropoutLayer(x)
        x = self.linear(x)

        return x
