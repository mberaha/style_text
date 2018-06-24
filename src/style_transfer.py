import json
import logging
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional
from collections import defaultdict
from src.base_model import BaseModel
from src.generate_batches import preprocessSentences
from src.rnn import Rnn
from src.discriminator import Cnn
from src.vocabulary import Vocabulary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StyleTransfer(BaseModel):

    def __init__(self, params, vocabulary: Vocabulary):
        super().__init__()
        self.vocabulary = vocabulary
        self.params = params
        # instantiating the encoder and the generator
        self.encoder = Rnn(
            params.autoencoder.input_size,
            params.autoencoder.hidden_size,
            params.autoencoder.num_layers,
            batch_first=True).to(device)
        self.generator = Rnn(
            params.autoencoder.input_size,
            params.autoencoder.hidden_size,
            params.autoencoder.num_layers,
            batch_first=True).to(device)

        # instantiating linear networks for hidden transformations
        self.labelsTransform = torch.nn.Linear(1, params.dim_y).to(device)
        self.hiddenToVocab = torch.nn.Linear(
            params.autoencoder.hidden_size,
            self.vocabulary.vocabSize).to(device)

        # instantiating the discriminators
        discriminator0 = Cnn(
            params.discriminator.in_channels,
            params.discriminator.out_channels,
            params.discriminator.kernel_sizes,
            params.autoencoder.hidden_size,
            params.discriminator.dropout
        ).to(device)
        discriminator1 = Cnn(
            params.discriminator.in_channels,
            params.discriminator.out_channels,
            params.discriminator.kernel_sizes,
            params.autoencoder.hidden_size,
            params.discriminator.dropout
        ).to(device)
        self.discriminators = {
            0: discriminator0,
            1: discriminator1
        }

        # instantiating the optimizer
        self.autoencoder_optimizer = optim.Adam(
            [{'params': self.encoder.parameters()},
             {'params': self.generator.parameters()},
             {'params': self.labelsTransform.parameters()},
             {'params': self.vocabulary.embeddings.parameters()},
             {'params': self.hiddenToVocab.parameters()}],
            lr=params.discriminator.learning_rate,
            betas=params.discriminator.betas)
        self.discriminator0_optimizer = optim.Adam(
            self.discriminators[0].parameters(),
            lr=params.discriminator.learning_rate,
            betas=params.discriminator.betas)
        self.discriminator1_optimizer = optim.Adam(
            self.discriminators[1].parameters(),
            lr=params.discriminator.learning_rate,
            betas=params.discriminator.betas)

        # instantiating the loss criterion
        self.rec_loss_criterion = nn.CrossEntropyLoss().to(device)
        self.adv_loss_criterion = nn.BCEWithLogitsLoss().to(device)

    def _encodeTokens(self, tokens, hiddens, lenghts):
        """
        This function takes as input a bach of lists of embeddings and returns
        the variable z: the encoded content
        Args:
        hidden -- h0
        """
        out, hidden = self.encoder(tokens, hiddens, lenghts)
        return hidden[:, :, self.params.dim_y:]

    def _generateTokens(self, tokens, h0, lenghts, evaluation):

        if evaluation:
            size = self.eval_size
        else:
            size = self.params.batch_size

        hidden = h0
        generatedVocabs = torch.zeros(
            size, len(tokens), self.vocabulary.vocabSize + 1,
            device=device)
        output, hidden = self.generator(tokens, hidden, lenghts)
        generatedVocabs = self.hiddenToVocab(output)
        return generatedVocabs, output

    def _generateWithPrevOutput(
            self, h0, max_len, lengths=[], evaluation=False, soft=True):
        """
        Implements professor teaching for the generator,transforming outputs
        at each time t to a curren token representing a weighted average
        of the embeddings (if soft=True) or just the most probable one (else)
        Params:
        h0 -- the first hidden state y + z of size (1, 1, hidden_size)
        max_len -- stops the generator after max_len tokens generated
        Output:
        hiddens -- of shape (max_len, 1, hidden_size).
                    If length<max_len it ends with zeros.
        """
        if evaluation:
            size = self.eval_size
        else:
            size = self.params.batch_size

        hidden = h0
        hiddens = torch.zeros(size, max_len,
                              self.params.autoencoder.hidden_size,
                              device=device)
        if soft:
            tokens = torch.zeros(
                size, max_len, self.params.embedding_size, device=device)
        else:
            tokens = torch.zeros(size, max_len, device=device)
        goEmbedding = self.vocabulary(['<go>']).squeeze(0)
        goEmbedding = goEmbedding.repeat(size, 1)
        goEmbedding = goEmbedding.unsqueeze(1)
        currTokens = goEmbedding

        for index in range(max_len):
            # generator need input (seq_len, batch_size, input_size)
            out, hidden = self.generator(
                currTokens, hidden, lengths, pad=False)
            vocabLogits = self.hiddenToVocab(out[:, 0, :])
            hiddens[:, index, :] = hidden

            # dropping some values of the logits during training
            if not evaluation:
                vocabLogits = torch.nn.functional.dropout(
                    vocabLogits, p=self.params.dropout)

            vocabProbs = nn.functional.softmax(
                vocabLogits / self.params.temperature, dim=1)
            if soft:
                currTokens = torch.matmul(
                    vocabProbs, self.vocabulary.embeddings.weight)
                tokens[:, index, :] = currTokens
            else:
                _, argmax = vocabProbs.max(1)
                tokens[:, index] = argmax
                currTokens = self.vocabulary(argmax)

            currTokens = currTokens.unsqueeze(1)

        hiddens = torch.cat((h0.transpose(0, 1), hiddens), dim=1)
        # tokens = torch.cat((goEmbedding, tokens), dim=1)
        return hiddens, tokens

    def adversarialLoss(self, x_real, x_fake, label):
        ones = torch.ones((len(x_real), 1)).to(device)
        discriminator = self.discriminators[label]
        x_fake = x_fake.unsqueeze(1)
        x_real = x_real.unsqueeze(1)
        class_fake = discriminator(x_fake)
        class_real = discriminator(x_real)
        class_fake = class_fake.squeeze(0)
        class_real = class_real.squeeze(0)

        labels = np.array([label] * x_real.shape[0])
        labels = torch.FloatTensor(labels).to(device)
        labels = labels.unsqueeze(1)
        loss_d = self.adv_loss_criterion(class_real, labels) + \
            self.adv_loss_criterion(class_fake, 1 - labels)
        loss_g = self.adv_loss_criterion(class_fake, ones)
        return loss_d, loss_g

    def _zeroGradients(self):
        self.autoencoder_optimizer.zero_grad()
        self.discriminator0_optimizer.zero_grad()
        self.discriminator1_optimizer.zero_grad()

    def _sentencesToInputs(self, sentences):
        # transform sentences into embeddings
        sentences = list(map(lambda x: x.split(" "), sentences))
        encoder_inputs, generator_inputs, targets, lengths = \
            preprocessSentences(sentences)
        encoder_inputs = torch.stack(list(map(
            self.vocabulary, encoder_inputs)))
        generator_inputs = torch.stack(list(map(
            self.vocabulary, generator_inputs)))
        targets = torch.stack(list(map(
            self.vocabulary.getSentenceIds, targets)))
        targets = nn.utils.rnn.pack_padded_sequence(
            targets, lengths, batch_first=True)[0]

        return encoder_inputs, generator_inputs, targets, lengths

    def printDebugLoss(self):
        out = {k: float(v) for k, v in self.losses.items()}
        logging.debug('Losses: \n{0}'.format(json.dumps(out, indent=4)))

    def _computeLosses(
            self, encoder_inputs, generator_inputs,
            targets, labels, lenghts, evaluation=False):
        self.losses = defaultdict(float)
        self._runBatch(
            encoder_inputs, generator_inputs,
            targets, labels, lenghts, evaluation)

    def _computeHiddens(
            self, encoder_inputs, generator_input, labels, lenghts, evaluation):
        if evaluation:
            size = self.eval_size
            labels = np.array(labels)
        else:
            size = self.params.batch_size

        tensorLabels = torch.FloatTensor(labels).to(device)
        tensorLabels = tensorLabels.unsqueeze(1)
        initialHiddens = self.labelsTransform(tensorLabels)
        initialHiddens = initialHiddens.unsqueeze(0)
        zeros = torch.zeros(1, size, self.params.dim_z, device=device)
        initialHiddens = torch.cat((initialHiddens, zeros), dim=2)
        content = self._encodeTokens(encoder_inputs, initialHiddens, lenghts)

        # generating the hidden states (yp, zp)
        originalHiddens = self.labelsTransform(tensorLabels)
        originalHiddens = originalHiddens.unsqueeze(0)
        self.originalHiddens = torch.cat(
            (originalHiddens, content), dim=2)

        # generating the hidden states with inverted labels (yq, zp)
        transformedHiddens = self.labelsTransform(1 - tensorLabels)
        transformedHiddens = transformedHiddens.unsqueeze(0)
        self.transformedHiddens = torch.cat(
            (transformedHiddens, content), dim=2)

    def _runBatch(
            self, encoder_inputs, generator_input,
            targets, labels, lenghts, evaluation):

        positiveIndex = np.nonzero(labels)
        negativeIndex = np.where(labels == 0)[0]

        self._computeHiddens(
                encoder_inputs, generator_input, labels, lenghts, evaluation)
        # reconstruction loss
        generatorOutputs, h_teacher = self._generateTokens(
            generator_input, self.originalHiddens, lenghts, evaluation)
        # re-pack padded sequence for computing losses
        packedGenOutput = nn.utils.rnn.pack_padded_sequence(
            generatorOutputs, lenghts, batch_first=True)[0]
        self.losses['reconstruction'] = self.rec_loss_criterion(
            packedGenOutput.view(-1, self.vocabulary.vocabSize),
            targets.view(-1))

        # adversarial losses
        h_professor, _ = self._generateWithPrevOutput(
            self.transformedHiddens, self.params.max_len,
            lenghts, evaluation, soft=True)

        # negative sentences
        d_loss, g_loss = self.adversarialLoss(
            h_teacher[negativeIndex],
            h_professor[positiveIndex],
            0)
        self.losses['discriminator0'] = d_loss
        self.losses['generator'] += g_loss

        # positive sentences
        d_loss, g_loss = self.adversarialLoss(
            h_teacher[positiveIndex],
            h_professor[negativeIndex],
            1)
        self.losses['discriminator1'] = d_loss
        self.losses['generator'] += g_loss

    @staticmethod
    def getNorm(parameters):
        # This is a debug function, to be removed when everything is working
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
        return total_norm ** (1. / 2)

    def trainOnBatch(self, sentences, labels, iterNum):
        self.train()
        labels = np.array(labels)
        encoder_inputs, generator_inputs, targets, lenghts = \
            self._sentencesToInputs(sentences)

        # print("trainOnBatch encoder_inputs.shape:", encoder_inputs.shape)
        # print("trainOnBatch generator_inputs.shape:", generator_inputs.shape)
        self._zeroGradients()
        self._computeLosses(
            encoder_inputs, generator_inputs, targets, labels, lenghts)

        self.losses['autoencoder'] = self.losses['reconstruction'] + \
            self.params.lambda_GAN * self.losses['generator']

        if iterNum % 500 == 0:
            self.printDebugLoss()

        self.losses['autoencoder'].backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(
            [*self.encoder.parameters(), *self.generator.parameters(),
             *self.labelsTransform.parameters(),
             *self.vocabulary.embeddings.parameters(),
             *self.hiddenToVocab.parameters()],
            self.params.grad_clip)

        self.autoencoder_optimizer.step()
        self._zeroGradients()

        self.losses['discriminator0'].backward(retain_graph=True)
        self.discriminator0_optimizer.step()
        self._zeroGradients()

        self.losses['discriminator1'].backward()
        self.discriminator1_optimizer.step()
        self._zeroGradients()
        return self.losses['autoencoder']

    def evaluateOnBatch(self, sentences, labels):
        self.eval()
        self.eval_size = len(sentences)
        encoder_inputs, generator_inputs, targets, lengths = \
            self._sentencesToInputs(sentences)

        self._computeLosses(
            encoder_inputs, generator_inputs,
            targets, labels, lengths, evaluation=True)

        self.losses['autoencoder'] = self.losses['reconstruction'] + \
            self.params.lambda_GAN * self.losses['generator']
        self.losses['autoencoder']

        self.losses['discriminator0']
        self.losses['discriminator1']

    def evaluate(self, batches):
        self.losses = defaultdict(float)
        for batch in batches:
            sentences = batch[0]
            labels = batch[1]
            self.evaluateOnBatch(sentences, labels)

        return self.losses['autoencoder']

    def transformBatch(self, sentences, labels):
        self.eval()
        self.eval_size = len(sentences)
        encoder_inputs, generator_inputs, _, lengths = \
            self._sentencesToInputs(sentences)
        self._computeHiddens(
            encoder_inputs, generator_inputs, labels, lengths, evaluation=True)
