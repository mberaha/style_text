import torch
import torch.nn as nn
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BeamState(object):

    def __init__(self, word, h, sentence, nll):
        """
        Args:
            word -- the id of the word charachterising the state
            h -- the hidden state associated to that state
            sentence -- a list of word ids (the past ids plus the current one)
            nll -- the negative log likelihood corresponding to the sentence
        """
        self.word, self.h, self.sentence, self.nll = \
            word, h, sentence, nll


class Decoder(object):

    def __init__(self, styleTransfer, max_len, beam_width, params):
        self.model = styleTransfer
        self.max_len = max_len
        self.width = beam_width
        self.params = params

        def _decode(self, h0s):

            batch_size = h0s.shape[1]
            hidden = h0s
            hiddens = torch.zeros(batch_size, max_len,
                                  self.params.autoencoder.hidden_size,
                                  device=device)
            tokens = torch.zeros(batch_size, max_len,
                                  params.embedding_size,
                                  device=device)
            goEmbedding = self.vocabulary(['<go>']).squeeze(0)
            goEmbedding = goEmbedding.repeat(batch_size, 1)
            goEmbedding = goEmbedding.unsqueeze(1)
            currTokens = goEmbedding

            for index in range(self.max_len):
                # generator need input (seq_len, batch_size, input_size)
                out, hidden = self.generator(currTokens, hidden, pad=False)
                vocabLogits = self.hiddenToVocab(out[:, 0, :])
                hiddens[:, index, :] = hidden

                vocabProbs = nn.functional.softmax(
                    vocabLogits / self.params.temperature, dim=1)
                _, argmax = vocabProbs.max(1)
                currTokens = self.vocabulary([argmax])
                tokens[:, index, :]

            hiddens = torch.cat((h0s.transpose(0, 1), hiddens), dim=1)
            tokens = torch.cat(goEmbedding, tokens, dim=1)
            return hiddens, tokens

    def rewriteBatch(self, sentences, labels):
        self.model.evaluateOnBatch(sentences, labels)
        originalHiddens = self.model.originalHiddens
        transformedHiddens = self.model.transformedHiddens
        original = self._decode(originalHiddens)
        transformed = self._decode(transformedHiddens)

        return original, transformed
