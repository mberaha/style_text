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
    return smoothed


class GaussianNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, din, stddev):
        if self.training:
            noise = torch.autograd.Variable(
                torch.randn(din.size()).to(device) * stddev)
            return din + noise
        return din


class StyleTransfer(BaseModel):

    def __init__(self, params, vocabulary: Vocabulary):
        super().__init__()
        self.vocabulary = vocabulary
        self.params = params
        self.discriminatorNoise = GaussianNoise()
        self.dropoutLayer = nn.Dropout(p=self.params.dropout)
        self.noise_sigma = self.params.initial_noise
        # instantiating the encoder and the generator
        self.encoder = Rnn(
            self.params.autoencoder.input_size,
            self.params.autoencoder.hidden_size,
            self.params.autoencoder.num_layers,
            batch_first=True,
            dropout=self.params.dropout).to(device)
        self.generator = Rnn(
            self.params.autoencoder.input_size,
            self.params.autoencoder.hidden_size,
            self.params.autoencoder.num_layers,
            batch_first=True,
            dropout=self.params.dropout).to(device)

        # instantiating linear networks for hidden transformations
        self.encoderLabelsTransform = \
            torch.nn.Linear(1, self.params.dim_y).to(device)
        self.generatorLabelsTransform = \
            torch.nn.Linear(1, self.params.dim_y).to(device)
        self.hiddenToVocab = torch.nn.Linear(
            self.params.autoencoder.hidden_size,
            self.vocabulary.vocabSize).to(device)

        # instantiating the discriminators
        discriminator0 = Cnn(
            self.params.discriminator.in_channels,
            self.params.discriminator.out_channels,
            self.params.discriminator.kernel_sizes,
            self.params.autoencoder.hidden_size,
            self.params.discriminator.dropout
        ).to(device)
        discriminator1 = Cnn(
            self.params.discriminator.in_channels,
            self.params.discriminator.out_channels,
            self.params.discriminator.kernel_sizes,
            self.params.autoencoder.hidden_size,
            self.params.discriminator.dropout
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
            lr=self.params.autoencoder.learning_rate,
            betas=(self.params.autoencoder.beta_0,
                   self.params.autoencoder.beta_1))
        self.discriminator0_optimizer = optim.Adam(
            self.discriminators[0].parameters(),
            lr=self.params.discriminator.learning_rate,
            betas=(self.params.discriminator.beta_0,
                   self.params.discriminator.beta_1))
        self.discriminator1_optimizer = optim.Adam(
            self.discriminators[1].parameters(),
            lr=self.params.discriminator.learning_rate,
            betas=(self.params.discriminator.beta_0,
                   self.params.discriminator.beta_1))

        # instantiating the loss criterion
        self.rec_loss_criterion = nn.CrossEntropyLoss().to(device)
        self.adv_loss_criterion = nn.BCEWithLogitsLoss().to(device)

    def adversarialLoss(self, x_real, x_fake, label, noisy=True):
        # initialize target tensors for the generator and the discriminator
        zeros = torch.zeros((len(x_fake), 1)).to(device)
        g_ones = torch.ones((len(x_real), 1)).to(device)
        d_ones = torch.ones((len(x_real), 1)).to(device)
        if self.params.discriminator.l_smoothing:
            d_ones = labelSmoothing(
                d_ones, self.params.discriminator.l_smoothing)
        if self.params.discriminator.l_flipping:
            d_ones = labelFlipping(
                d_ones, self.params.discriminator.l_flipping)

        # choose which discriminator to apply
        discriminator = self.discriminators[label]
        # prepare discriminator's inputs
        x_real = x_real.unsqueeze(1)
        x_fake = x_fake.unsqueeze(1)
        if noisy:
            x_real = self.discriminatorNoise(x_real, self.noise_sigma)
            x_fake = self.discriminatorNoise(x_fake, self.noise_sigma)

        # run discriminator
        class_fake = discriminator(x_fake.detach())
        class_real = discriminator(x_real.detach())
        class_fake = class_fake.squeeze(0)
        class_real = class_real.squeeze(0)

        # calculate adversarial loss for d
        loss_d = self.adv_loss_criterion(class_real, d_ones) + \
            self.adv_loss_criterion(class_fake, zeros)
        # calculate non-saturating loss for g (see Goodfellow 2014)
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

        hidden = h0
        hiddens = torch.zeros(
            self.size, max_len, self.params.autoencoder.hidden_size,
            device="cuda")
        if soft:
            tokens = torch.zeros(
                self.size, max_len, self.params.embedding_size, device="cuda")
        else:
            tokens = torch.zeros(self.size, max_len, device="cuda")

        goEmbedding = self.vocabulary(['<go>']).squeeze(0)
        goEmbedding = goEmbedding.repeat(self.size, 1)
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
                currTokens = currTokens.unsqueeze(1)
                hiddens[:, index, :] = hidden

        else:
            for index in range(max_len):
                output, hidden = self.generator(currTokens, hidden, pad=False)
                vocabLogit = self.hiddenToVocab(hidden)
                idxs = vocabLogit[0, :, :].max(1)[1]
                tokens[:, index] = idxs
                currTokens = self.vocabulary(idxs, byWord=False).unsqueeze(1)

        hiddens = torch.cat((h0.transpose(0, 1), hiddens), dim=1)
        return hiddens, tokens

    def _generateTokens(self, tokens, h0, lenghts, evaluation):
        hidden = h0
        generatedVocabs = torch.zeros(
            self.size, len(tokens), self.vocabulary.vocabSize + 1,
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

        self.size = self.eval_size if evaluation else self.params.batch_size
        self.losses = defaultdict(float)
        negativeIndex = np.where(labels == 0)[0]
        positiveIndex = np.nonzero(labels)

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
            preprocessSentences(
                sentences, noisy=noisy,
                word_drop=self.params.autoencoder.word_drop)
        encoder_inputs = torch.autograd.Variable(torch.stack(list(map(
            self.vocabulary, encoder_inputs))))
        generator_inputs = torch.autograd.Variable(torch.stack(list(map(
            self.vocabulary, generator_inputs))))
        targets = torch.autograd.Variable(torch.stack(list(map(
            self.vocabulary.getSentenceIds, targets))))
        targets = nn.utils.rnn.pack_padded_sequence(
            targets, lengths, batch_first=True)

        return encoder_inputs, generator_inputs, targets, lengths

    def trainOnBatch(self, sentences, labels, iterNum):
        self.train()
        labels = np.array(labels)
        encoder_inputs, generator_inputs, targets, lenghts = \
            self._sentencesToInputs(
                sentences, noisy=True)

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

        labels = np.array(labels)
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
            reconstructedGreedy = []
            transformedGreedy = []
            reconstructedBeam = []
            transformedBeam = []
            allSentences = []
            allLabels = []
            epoch_dir = os.path.join(
                self.params.logdir, 'epoch_{0}'.format(epoch_index))
            os.makedirs(epoch_dir, exist_ok=True)
            lossFile = os.path.join(epoch_dir, 'losses.pickle')
            transferFile = os.path.join(epoch_dir, 'transfers.json')
            with open(lossFile, 'wb') as fp:
                pickle.dump(batchLosses, fp)

            for batch in batches:
                # rnn sorts sentences from longest to shortest, to keep the
                # correspondance we do the same.
                allSentences.extend(sorted(batch[0], key=len, reverse=True))
                allLabels.extend(batch[1])
                rGreedy, tGreedy = greedy.rewriteBatch(batch[0], batch[1])
                reconstructedGreedy.extend(rGreedy)
                transformedGreedy.extend(tGreedy)
                rBeam, tBeam = beam.rewriteBatch(batch[0], batch[1])
                reconstructedBeam.extend(rBeam)
                transformedBeam.extend(tBeam)

            with open(transferFile, 'w') as fp:
                json.dump(
                    {'original': allSentences,
                     'labels': allLabels,
                     'reconstructed_greedy': reconstructedGreedy,
                     'transformed_greedy': transformedGreedy,
                     'reconstructed_beam': reconstructedBeam,
                     'transformed_beam': transformedBeam}, fp)

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
