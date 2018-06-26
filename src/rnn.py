import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def SoftSampleWord(dropout, embeddings, gamma):
    """
    Given the output of the generator, performs a dropout over it
    and then apply the Gumbel_softmax trick
    """
    def GumbelSoftmax(logits, gamma, eps=1e-20):
        U = torch.rand(logits.shape).to(device)
        G = -torch.log(-torch.log(U + eps) + eps)
        return nn.functional.softmax(
            (logits + G) / gamma, dim=1)  # log(logits) is better???

    def loop_func(output, hiddenToVocab):
        out = torch.nn.functional.dropout(output, p=dropout)
        vocabLogits = hiddenToVocab(out[:, 0, :])
        vocabProbs = GumbelSoftmax(vocabLogits, gamma)
        currTokens = torch.matmul(
            vocabProbs, embeddings.weight)
        return currTokens, vocabLogits

    return loop_func
