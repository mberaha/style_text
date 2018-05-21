import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from src.generate_batches import preprocessSentences
from src.rnn import Rnn
from src.vocabulary import Vocabulary
from src.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleTransfer(object):

    def __init__(self, params, vocabulary: Vocabulary):

        # instantiating the encoder and the generator
        self.encoder = Rnn(
            params.autoencoder.input_size,
            params.autoencoder.hidden_size,
            params.autoencoder.num_layers,
            device)
        self.generator = Rnn(
            params.autoencoder.input_size,
            params.autoencoder.hidden_size,
            params.autoencoder.num_layers,
            device)

        # instantiating the solvers
        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(),
            lr=params.autoencoder.learning_rate,
            betas=params.autoencoder.betas)
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=params.autoencoder.learning_rate,
            betas=params.autoencoder.betas)

        # instantiating the loss criterion
        self.loss_criterion = nn.CrossEntropyLoss()

        # instantiating the vocabulary
        self.vocabulary = vocabulary

        # instantiating some useful functions
        self.labelsTransform = torch.nn.Linear(1, params.dim_y)
        self.hiddenToVocab = torch.nn.Linear(
            params.hidden_size, self.vocabulary.vocabSize + 1)

    def _encodeTokens(self, tokens, hidden):
        """
        This function takes as input a list of embeddings and returns
        the variable z: the encoded content
        """
        z = [] # TODO verificare che non ci siano casini appendendo tensori a z
        for token in tokens:
            out, hidden = self.encoder(token, hidden)
            z.append(hidden[:, :, self.params.dim_y:])
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
        labels = np.array(labels)
        encoder_inputs, decoder_inputs, targets = preprocessSentences(sentences)
        encoder_inputs = list(map(self.vocabulary.getEmbedding, encoder_inputs))
        decoder_inputs = list(map(self.vocabulary.getEmbedding, decoder_inputs))
        targets = list(map(self.vocabulary.getWordId, sentences))

        self.losses = defaultdict(float)
        originalHidden = []
        transformedHidden = []
        for index, sentence in enumerate(sentences):

            #####   auto-encoder   #####
            self.encoder_optimizer.zero_grad()
            # initialize the first hidden state of the encoder
            initialHidden = self.labelsTransform(labels[index])
            initialHidden = initialHidden.unsqueeze(0).unsqueeze(0)
            initialHidden = torch.cat(
                (initialHidden, torch.zeros(1, 1, params.dim_z)), dim=2)

            # encode tokens and extract only content=hidden[:,:,dim_y:]
            content = self._encodeTokens(sentence, initialHidden)

            # TODO continue here checking all the tensors
            originalHidden = self.labelsTransform(labels[index])
            originalHidden = originalHidden.unsqueeze(0).unsqueeze(0)
            originalHidden = torch.cat(
                (originalHidden, content), dim=2)
            transformedHidden = self.labelsTransform(1 - labels[index])
            transformedHidden.extend(content)

            self.generator_optimizer.zero_grad()

            generatorOutput = self._generateTokens(sentence, originalHidden)
            vocabOutput = self.hiddenToVocab(generatorOutput)
            self.losses['reconstruction'] += self.reconstructionLoss(
                vocabOutput, targets)
