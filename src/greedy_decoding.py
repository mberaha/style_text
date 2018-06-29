import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GreedyDecoder(object):

    def __init__(self, styleTransfer, max_len, beam_width, params):
        self.model = styleTransfer
        self.max_len = max_len
        self.width = beam_width
        self.params = params

    def _decode(self, h0s):
        batch_size = h0s.shape[1]
        hiddens, tokens = self.model._generateWithPrevOutput(
            h0s, self.max_len, evaluation=True, soft=False)
        sentences = []
        for i in range(batch_size):
            curr = tokens[i, :]
            sent = [self.model.vocabulary.id2word[int(x)] for x in list(curr)]
            sentences.append(" ".join(sent))
        return sentences

    def rewriteBatch(self, sentences, labels):
        self.model.transformBatch(sentences, labels)
        originalHiddens = self.model.originalHiddens
        transformedHiddens = self.model.transformedHiddens
        original = self._decode(originalHiddens)
        transformed = self._decode(transformedHiddens)

        return original, transformed
