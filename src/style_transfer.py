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
            params.hidden_size, self.vocabulary.vocabSize + 1)

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

    def _generateWithPrevOutput(self, hidden, length, soft=True):
        hiddens = []
        allLogits = []
        currToken = self.vocabulary.embeddings['<go>']
        softmax = torch.nn.Softmax()
        for index in range(length):
            hiddens.append(hidden)
            out, hidden = self.generator(currToken, hidden)
            vocabLogits = self.hiddenToVocab(out)
            allLogits.append(vocabLogits)
            # TODO add dropout
            vocabProbs = softmax(vocabLogits / self.params.temperature)
            if soft:
                currToken = torch.matmul(vocabProbs, self.vocabulary.embeddings)
            else:
                _, argmax = vocabProbs.max(1)
                currToken = self.vocabulary.embeddings[argmax]

        return hiddens, allLogits

    def reconstructionLoss(self, outputs, targets):
        return torch.nn.functional.cross_entropy(outputs, targets)

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
            vocabOutput = self.hiddenToVocab(generatorOutput)
            self.losses['reconstruction'] += self.reconstructionLoss(
                vocabOutput, targets)
