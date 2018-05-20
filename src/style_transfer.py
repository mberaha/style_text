import numpy as np
import torch
from collections import defaultdict
from src.generate_batches import preprocessSentences
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
        self.labelsTransform = torch.nn.Linear(1, params.dim_y)
        self.hiddenToVocab = torch.nn.Linear(
            params.hidden_size, self.vocabulary.vocabSize)

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
            out, hidden = self.generator(token, hidden)
            generatedOutputs.append(out)
        return generatedOutputs

    def reconstructionLoss(self, generatorOutput, targets):
        wordOutputs = self.hiddenToVocab(generatorOutput)
        return torch.nn.functional.cross_entropy(wordOutputs, targets)

    def trainOnBatch(self, sentences, labels):
        # transform sentences into embeddings
        self.losses = defaultdict(float)
        labels = np.array(labels)
        encoder_inputs, decoder_inputs, targets = preprocessSentences(sentences)
        encoder_inputs = list(map(self.vocabulary.getEmbedding, encoder_inputs))
        decoder_inputs = list(map(self.vocabulary.getEmbedding, decoder_inputs))
        targets = list(map(self.vocabulary.getWordId, sentences))
        originalHidden = []
        transformedHidden = []
        for index, sentence in enumerate(sentences):
            initialHidden = self.labelsTransform(labels[index])
            initialHidden.extend(torch.zeros(1, 1, self.params.dim_z))

            content = self._encodeTokens(sentence, initialHidden)

            originalHidden = self.labelsTransform(labels[index])
            originalHidden.extend(content)
            transformedHidden = self.labelsTransform(1 - labels[index])
            transformedHidden.extend(content)

            generatorOutput = self._generateTokens(sentence, originalHidden)
            self.losses['reconstruction'] += self.reconstructionLoss(
                generatorOutput, targets)
