import copy
import json
import logging
import numpy as np
import os
import pickle
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional
from torch.autograd.variable import Variable
from collections import defaultdict
from src.beam_search import BeamSearchDecoder
from src.greedy_decoding import GreedyDecoder
from src.base_model import BaseModel
from src.generate_batches import preprocessSentences
from src.rnn import Rnn, SoftSampleWord
from src.discriminator import Cnn
from src.vocabulary import Vocabulary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def labelFlipping(ones, flipping):
    """
    apply one-sided label flipping to positive labels as described in:
    https://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    """
    s = ones.shape[0]
    probs = ones / s
    nSamples = int(round(s * flipping))
    indexToFlip = torch.multinomial(probs[:, 0], nSamples)
    ones[indexToFlip] = 0
    return ones


def labelSmoothing(ones, smoothing):
    """
    apply one sided smoothing to positive labels as described in:
    Improved Techniques for Training GANs - 2016
    """
    randVec = torch.rand(ones.shape).to(device)
    smoothed = ones - randVec*smoothing
    return Variable(smoothed)


class GaussianNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, din, stddev):
        if self.training:
            noise = Variable(
                torch.randn(din.size()).to(device) * stddev)
            return din + noise
        return din


class StyleTransfer(BaseModel):

    def __init__(self, params, vocabulary: Vocabulary):
        super().__init__()
        self.vocabulary = vocabulary
        self.params = params
        self.dropoutLayer = nn.Dropout(p=self.params.dropout)
        self.discriminatorNoise = GaussianNoise()
        self.noise_sigma = self.params.initial_noise
        # instantiating the encoder and the generator
        self.encoder = Rnn(
            params.autoencoder.input_size,
            params.autoencoder.hidden_size,
            params.autoencoder.num_layers,
            batch_first=True,
            dropout=self.params.dropout).to(device)
        self.generator = Rnn(
            params.autoencoder.input_size,
            params.autoencoder.hidden_size,
            params.autoencoder.num_layers,
            batch_first=True,
            dropout=self.params.dropout).to(device)

        # instantiating linear networks for hidden transformations
        self.encoderLabelsTransform = \
            torch.nn.Linear(1, params.dim_y).to(device)
        self.generatorLabelsTransform = \
            torch.nn.Linear(1, params.dim_y).to(device)
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
             {'params': self.encoderLabelsTransform.parameters()},
             {'params': self.generatorLabelsTransform.parameters()},
             {'params': self.vocabulary.embeddings.parameters()},
             {'params': self.hiddenToVocab.parameters()}],
            lr=params.discriminator.learning_rate,
            betas=(self.params.autoencoder.beta_0,
                   self.params.autoencoder.beta_1))
        self.discriminator0_optimizer = optim.Adam(
            self.discriminators[0].parameters(),
            lr=params.discriminator.learning_rate,
            betas=(self.params.autoencoder.beta_0,
                   self.params.autoencoder.beta_1))
        self.discriminator1_optimizer = optim.Adam(
            self.discriminators[1].parameters(),
            lr=params.discriminator.learning_rate,
            betas=(self.params.autoencoder.beta_0,
                   self.params.autoencoder.beta_1))

        # instantiating the loss criterion
        self.rec_loss_criterion = nn.CrossEntropyLoss().to(device)
        self.adv_loss_criterion = nn.BCEWithLogitsLoss().to(device)

    def adversarialLoss(self, x_real, x_fake, label, noisy: bool):
        d_ones = Variable(torch.ones((len(x_real), 1)).to(device))
        g_ones = Variable(torch.ones((len(x_real), 1)).to(device))
        zeros = Variable(torch.zeros((len(x_fake), 1)).to(device))
        if self.params.discriminator.l_smoothing:
            d_ones = labelSmoothing(
                d_ones, self.params.discriminator.l_smoothing)
        if self.params.discriminator.l_flipping:
            d_ones = labelFlipping(
                d_ones, self.params.discriminator.l_flipping)

        discriminator = self.discriminators[label]
        x_fake = x_fake.unsqueeze(1)
        x_real = x_real.unsqueeze(1)
        if noisy:
            x_real = self.discriminatorNoise(x_real, self.noise_sigma)
            x_fake = self.discriminatorNoise(x_fake, self.noise_sigma)
        class_fake = discriminator(x_fake.detach())
        class_real = discriminator(x_real.detach())
        class_fake = class_fake.squeeze(0)
        class_real = class_real.squeeze(0)

        loss_d = self.adv_loss_criterion(class_real, d_ones) + \
            self.adv_loss_criterion(class_fake, zeros)
        loss_g = self.adv_loss_criterion(class_fake, g_ones)
        return loss_d, loss_g

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
        softSampleFunction = SoftSampleWord(
            dropout=self.params.dropout,
            embeddings=self.vocabulary.embeddings,
            gamma=self.params.temperature)

        if soft:
            for index in range(max_len):
                # generator need input (seq_len, batch_size, input_size)
                output, hidden = self.generator(
                    currTokens, hidden, pad=False)
                currTokens, vocabLogits = softSampleFunction(
                    output=output,
                    hiddenToVocab=self.hiddenToVocab)
                tokens[:, index, :] = currTokens
                hiddens[:, index, :] = hidden
                currTokens = currTokens.unsqueeze(1)

        else:
            for index in range(max_len):
                output, hidden = self.generator(currTokens, hidden, pad=False)
                hidden = self.dropoutLayer(hidden)
                vocabLogits = self.hiddenToVocab(hidden)
                idxs = vocabLogits[0, :, :].max(1)[1]
                tokens[:, index] = idxs
                currTokens = self.vocabulary(idxs, byWord=False).unsqueeze(1)

        # hiddens = torch.cat((h0.transpose(0, 1), hiddens), dim=1)
        # tokens = torch.cat((goEmbedding, tokens), dim=1)
        return hiddens, tokens

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

        # dropping some values of the generator output
        # during both training and test
        output = self.dropoutLayer(output)

        generatedVocabs = self.hiddenToVocab(output)
        return generatedVocabs, output

    def _computeHiddens(
            self, encoder_inputs, generator_input, labels, lenghts, evaluation):
        if evaluation:
            size = self.eval_size
            labels = np.array(labels)
        else:
            size = self.params.batch_size

        tensorLabels = torch.FloatTensor(labels).to(device)
        tensorLabels = tensorLabels.unsqueeze(1)
        initialHiddens = self.encoderLabelsTransform(tensorLabels)
        initialHiddens = initialHiddens.unsqueeze(0)
        zeros = torch.zeros(1, size, self.params.dim_z, device=device)
        initialHiddens = torch.cat((initialHiddens, zeros), dim=2)
        _, content = self.encoder(encoder_inputs, initialHiddens, lenghts)
        content = content[:, :, self.params.dim_y:]

        # generating the hidden states (yp, zp)
        originalHiddens = self.generatorLabelsTransform(tensorLabels)
        originalHiddens = originalHiddens.unsqueeze(0)
        self.originalHiddens = torch.cat(
            (originalHiddens, content), dim=2)

        # generating the hidden states with inverted labels (yq, zp)
        transformedHiddens = self.generatorLabelsTransform(1 - tensorLabels)
        transformedHiddens = transformedHiddens.unsqueeze(0)
        self.transformedHiddens = torch.cat(
            (transformedHiddens, content), dim=2)

    def _runBatch(
            self, encoder_inputs, generator_input,
            targets, labels, lenghts, evaluation, which_params):
        """
        which_params - string 'd0' or 'd1' or 'eg'
        """

        self.losses = defaultdict(float)

        negativeIndex = [
            i for i in range(self.params.batch_size//2)]
        positiveIndex = [
            i for i in range(self.params.batch_size//2, self.params.batch_size)]

        self._computeHiddens(
                encoder_inputs, generator_input, labels, lenghts, evaluation)

        generatorOutputs, h_teacher = self._generateTokens(
            generator_input, self.originalHiddens, lenghts, evaluation)

        # econder and generator's losses
        if which_params == 'eg':
            # re-pack padded sequence for computing losses
            packedGenOutput = nn.utils.rnn.pack_padded_sequence(
                generatorOutputs, lenghts, batch_first=True)[0]

            self.losses['reconstruction'] = self.rec_loss_criterion(
                packedGenOutput.view(-1, self.vocabulary.vocabSize),
                targets[0].view(-1))

        # adversarial losses
        h_professor, _ = self._generateWithPrevOutput(
            self.transformedHiddens, self.params.max_len,
            lenghts, evaluation, soft=True)

        if which_params in ['eg', 'd0']:
            d_loss, g_loss = self.adversarialLoss(
                h_teacher[negativeIndex],
                h_professor[positiveIndex],
                0, noisy=not evaluation)
            self.losses['discriminator0'] = d_loss
            self.losses['generator'] += g_loss

        # positive sentences
        if which_params in ['eg', 'd1']:
            d_loss, g_loss = self.adversarialLoss(
                h_teacher[positiveIndex],
                h_professor[negativeIndex],
                1, noisy=not evaluation)
            self.losses['discriminator1'] = d_loss
            self.losses['generator'] += g_loss

    def _zeroGradients(self):
        self.autoencoder_optimizer.zero_grad()
        self.discriminator0_optimizer.zero_grad()
        self.discriminator1_optimizer.zero_grad()

    def _sentencesToInputs(self, sentences, noisy):
        # transform sentences into embeddings
        sentences = list(map(lambda x: x.split(" "), sentences))
        encoder_inputs, generator_inputs, targets, lengths = \
            preprocessSentences(sentences, noisy=noisy)
        encoder_inputs = torch.stack(list(map(
            self.vocabulary, encoder_inputs)))
        generator_inputs = torch.stack(list(map(
            self.vocabulary, generator_inputs)))
        targets = torch.stack(list(map(
            self.vocabulary.getSentenceIds, targets)))
        targets = nn.utils.rnn.pack_padded_sequence(
            targets, lengths, batch_first=True)

        return encoder_inputs, generator_inputs, targets, lengths

    def trainOnBatch(self, sentences, labels, iterNum):
        self.train()
        labels = np.array(labels)
        encoder_inputs, generator_inputs, targets, lenghts = \
            self._sentencesToInputs(sentences, noisy=True)

        # compute losses for discriminator0 and optimize
        self._zeroGradients()
        self._runBatch(
            encoder_inputs, generator_inputs, targets, labels, lenghts,
            evaluation=False, which_params='d0')

        d0Loss = self.losses['discriminator0']
        self.losses['discriminator0'].backward()
        self.discriminator0_optimizer.step()

        # compute losses for discriminator1 and optimize
        self._zeroGradients()
        self._runBatch(
            encoder_inputs, generator_inputs, targets, labels, lenghts,
            evaluation=False, which_params='d1')

        d1Loss = self.losses['discriminator1']
        self.losses['discriminator1'].backward()
        self.discriminator1_optimizer.step()

        # compute losses for encoder and generator and optimize
        self._zeroGradients()
        self._runBatch(
            encoder_inputs, generator_inputs, targets, labels, lenghts,
            evaluation=False, which_params='eg')
        self.losses['autoencoder'] = self.losses['reconstruction'].clone()
        if d1Loss < self.params.max_d_loss and \
                d0Loss < self.params.max_d_loss:
            self.losses['autoencoder'] += \
                self.params.lambda_GAN * self.losses['generator']

        if iterNum % 200 == 0:
            self.losses['discriminator1'] = d1Loss
            self.losses['discriminator0'] = d0Loss
            self.printDebugLoss()

        self.losses['autoencoder'].backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(
            [*self.encoder.parameters(), *self.generator.parameters(),
             *self.encoderLabelsTransform.parameters(),
             *self.generatorLabelsTransform.parameters(),
             *self.vocabulary.embeddings.parameters(),
             *self.hiddenToVocab.parameters()],
            self.params.grad_clip)

        self.autoencoder_optimizer.step()
        self.noise_sigma = self.noise_sigma * self.params.noise_decay
        return self.losses['autoencoder']

    def evaluateOnBatch(self, sentences, labels):
        self.eval()
        self.eval_size = len(sentences)
        encoder_inputs, generator_inputs, targets, lengths = \
            self._sentencesToInputs(sentences, noisy=False)

        self._runBatch(
            encoder_inputs, generator_inputs, targets, labels, lengths,
            evaluation=True, which_params='eg')

        self.losses['autoencoder'] = self.losses['reconstruction'] + \
            self.params.lambda_GAN * self.losses['generator']

        return self.losses['autoencoder']

    def evaluate(self, batches, epoch_index):
        batchLosses = []
        self.losses = defaultdict(float)
        for batch in batches:
            sentences = batch[0]
            labels = batch[1]
            self.evaluateOnBatch(sentences, labels)
            batchLosses.append(
                {k: copy.deepcopy(float(v)) for k, v in self.losses.items()})

        if self.params.logdir:
            greedy = GreedyDecoder(self, self.params)
            beam = BeamSearchDecoder(self, self.params)
            epoch_dir = os.path.join(
                self.params.logdir, 'epoch_{0}'.format(epoch_index))
            os.makedirs(epoch_dir, exist_ok=True)
            lossFile = os.path.join(epoch_dir, 'losses.pickle')
            transferFile = os.path.join(epoch_dir, 'transfers.json')
            with open(lossFile, 'wb') as fp:
                pickle.dump(batchLosses, fp)

            inputs, labels = batches[0]
            rGreedy, tGreedy = greedy.rewriteBatch(inputs, labels)
            rBeam, tBeam = beam.rewriteBatch(inputs, labels)

            encoder_inputs, generator_inputs, targets, lenghts = \
                self._sentencesToInputs(inputs, noisy=False)
            self._computeHiddens(
                    encoder_inputs, generator_inputs, labels, lenghts, True)
            reconstructed, _ = self._generateTokens(
                generator_inputs, self.originalHiddens, lenghts, True)
            reconstructedIds = reconstructed.max(2)[1]
            reconstructedSents = []
            for i in range(reconstructedIds.shape[0]):
                ids = reconstructedIds[i, :]
                reconstructedSents.append(
                    " ".join([self.vocabulary.id2word[x] for x in ids]))

            with open(transferFile, 'w') as fp:
                json.dump(
                    {'labels': batch[1],
                     'reconstructed': reconstructedSents,
                     'reconstructed_greedy': rGreedy,
                     'transformed_greedy': tGreedy,
                     'reconstructed_beam': rBeam,
                     'transformed_beam': tBeam}, fp)

        return np.average([float(x['autoencoder']) for x in batchLosses])

    def transformBatch(self, sentences, labels):
        self.eval()
        self.eval_size = len(sentences)
        encoder_inputs, generator_inputs, _, lengths = \
            self._sentencesToInputs(sentences, noisy=False)
        self._computeHiddens(
            encoder_inputs, generator_inputs, labels, lengths, evaluation=True)

    def printDebugLoss(self):
        out = {k: float(v) for k, v in self.losses.items()}
        logging.debug('Losses: \n{0}'.format(json.dumps(out, indent=4)))

    @staticmethod
    def getNorm(parameters):
        # This is a debug function, to be removed when everything is working
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            return total_norm ** (1. / 2)
