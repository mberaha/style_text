import numpy as np
import torch
from src.rnn import Rnn
from src.vocabulary import Vocabulary


class StyleTransfer(object):

    def __init__(self, params, vocabulary: Vocabulary):
        self.vocabulary = vocabulary
        self.encoder = Rnn(
            params.encoder.input_size,
            params.encoder.hidden_size,
            params.encoder.num_layers)
        self.generator = Rnn(
            params.generator.input_size,
            params.generator.hidden_size,
            params.generator.num_layers)
        self.params = params
        self.labelsTransform = torch.nn.Linear(params.batch_size, params.dim_y)

    def _encodeTokens(self, tokens, hidden):
        """
        This function takes as input a list of embeddings and returns
        the variable z: the encoded content
        """
        z = []
        for token in tokens:
            out, hidden = self.encoder(token, hidden)
            z.append(hidden[self.params.dim_y:])
        return z

    def _generateTokens(self, tokens, hidden):
        generatedOutputs = []
        for token in tokens:
            out, hidden = self.generator(hidden)
            generatedOutputs.append(out)
        return generatedOutputs

    def reconstructionLoss(self, outputs, targets):
        return torch.nn.functional.cross_entropy(outputs, targets)

    def trainOnBatch(self, sentences, labels):
        # transform sentences into embeddings
        labels = np.array(labels)
        sentences = list(map(self.vocabulary.getEmbedding, sentences))
        initialHidden = self.encoder.initHidden()
        contents = []
        # encode
        for sentence in sentences:
            contents.append(self._encodeTokens(sentence, initialHidden))

        originalHidden = torch.LongTensor(
            [self.labelsTransform(labels)].extend(contents))
        transformedHidden = torch.LongTensor(
            [self.labelsTransform(1 - labels)].extend(contents))
        # decode
        generatorOutputs = []
        for index, sentence in enumerate(sentences):
            generatorOutputs.append(
                self._generateTokens(sentence, originalHidden[index]))
