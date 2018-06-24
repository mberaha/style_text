import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(object):

    def __init__(self, styleTransfer, max_len, beam_width, params):
        self.model = styleTransfer
        self.max_len = max_len
        self.width = beam_width
        self.params = params

    def _decodeModel(self, h0s):
        batch_size = h0s.shape[1]
        hiddens, tokens = self.model._generateWithPrevOutput(
            h0s, self.max_len, evaluation=True, soft=False)
        sentences = []
        for sentIndex in range(batch_size):
            sent = tokens[sentIndex, :]
            sentences.append(
                [self.model.vocabulary.id2word[int(sent[i])]
                 for i in range(self.max_len)])
        return sentences

    def _decode(self, h0s):

        batch_size = h0s.shape[1]
        hidden = h0s
        tokens = torch.zeros(batch_size, self.max_len, device=device)
        goEmbedding = self.model.vocabulary(['<go>']).squeeze(0)
        goEmbedding = goEmbedding.repeat(batch_size, 1)
        goEmbedding = goEmbedding.unsqueeze(1)
        currTokens = goEmbedding

        for index in range(self.max_len):
            # generator need input (seq_len, batch_size, input_size)
            out, hidden = self.model.generator(currTokens, hidden, pad=False)
            print('out.shape ', out.shape)
            # vocabLogits = self.model.hiddenToVocab(out[:, 0, :])
            vocabLogits = self.model.hiddenToVocab(out)
            # hiddens[:, index, :] = hidden
            print("vocabLogits.shape: ", vocabLogits.shape)

            _, argmax = vocabLogits[:, 0, :].max(1)
            tokens[:, index] = argmax
            currTokens = self.model.vocabulary(argmax)
            currTokens = currTokens.unsqueeze(1)

        sentences = []
        for sentIndex in range(batch_size):
            sent = tokens[sentIndex, :]
            sentences.append(
                [self.model.vocabulary.id2word[int(sent[i])]
                 for i in range(self.max_len)])
        return sentences

    def rewriteBatch(self, sentences, labels):
        self.model.transformBatch(sentences, labels)
        originalHiddens = self.model.originalHiddens
        transformedHiddens = self.model.transformedHiddens
        original = self._decodeModel(originalHiddens)
        transformed = self._decodeModel(transformedHiddens)

        return original, transformed
