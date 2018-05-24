import torch
import torch.nn as nn
import torch.nn.functional as F


class Cnn(nn.Module):
    """
        GAN discriminator is a CNN
    """

    def __init__(self, in_channels, out_channels, kernel_sizes, emb_size,
                 hidden_size, dropout):
        """
        Args:
        in_channels -- the input feature maps. Should be only one for text.
        out_channels -- the output feature maps a.k.a response maps
                        = number of filters/kernels
        kernel_sizes -- the lengths of the filters. Should be the number of
                        embeddings sweeped at a time by the different filters.
        emb_size -- size of the embedding
        hidden_size = is the number of hidden units for the Linear layer

        """

        super().__init__()

        # build parallel CNNs with different kernel sizes
        self.convs = nn.ModuleList([])  # CHECK CONVS ON ORIGINAL PAPER
        for ks in kernel_sizes:
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, (ks, emb_size)))

        self.linear = nn.Linear(out_channels*len(kernel_sizes), 1)

    def forward(self, x):
        """
        Args:
        x -- (batch_size * in_channels=1 * seq_len * emb_dim)
        """

        convs = [
            F.leaky_relu(conv(x), negative_slope=0.01) for conv in self.convs]
        pools = [
            F.max_pool2d(conv, (conv.size()[2], 1)) for conv in convs]
        flatten = torch.cat(pools, dim=1).view(x.size()[0], -1)
        logits = F.dropout(F.relu(self.linear(flatten)))

        return logits
