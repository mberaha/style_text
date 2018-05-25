import numpy as np
import torch
from torch import optim
import torch.nn as nn
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

        # instantiating linear networks for hidden transformations
        self.labelsTransform = torch.nn.Linear(1, params.dim_y)
        self.hiddenToVocab = torch.nn.Linear(
            params.autoencoder.hidden_size, self.vocabulary.vocabSize + 1)

        # instantiating the discriminators
        discriminator0 = Cnn(
            params.discriminator.in_channels,
            params.discriminator.out_channels,
            params.discriminator.kernel_sizes,
            params.discriminator.embedding_size,
            params.discriminator.hidden_size,
            params.discriminator.dropout
        )
        discriminator1 = Cnn(
            params.discriminator.in_channels,
            params.discriminator.out_channels,
            params.discriminator.kernel_sizes,
            params.discriminator.embedding_size,
            params.discriminator.hidden_size,
            params.discriminator.dropout
        )
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
            lr=params.autoencoder.learning_rate,
            betas=params.autoencoder.betas)
        self.discriminator0_optimizer = optim.Adam(
            self.discriminators[0].parameters(),
            lr=params.discriminator.learning_rate,
            betas=params.discriminator.betas)
        self.discriminator1_optimizer = optim.Adam(
            self.discriminators1.parameters(),
            lr=params.discriminator.learning_rate,
            betas=params.discriminator.betas)

        # instantiating the loss criterion
        self.rec_loss_criterion = nn.CrossEntropyLoss()
        self.adv_loss_criterion = nn.BCEWithLogitsLoss()

    def _encodeTokens(self, tokens, hidden):
        """
        This function takes as input a list of embeddings and returns
        the variable z: the encoded content
        Args:
        hidden -- h0
        """
        for token in tokens:
            out, hidden = self.encoder(token, hidden)
        return hidden[:, :, self.params.dim_y:]

    def _generateTokens(self, tokens, h0):
        hidden = h0
        generatedVocabs = torch.zeros(
            len(tokens), self.vocabulary.vocabSize + 1, device=device)
        output, hidden = self.generator(tokens, hidden)
        for i in range(output.shape[0]):
            curr = output[i, 0, :]
            generatedVocabs[i, :] = self.hiddenToVocab(curr)
        return generatedVocabs, output

    def _generateWithPrevOutput(self, h0, max_length, soft=True):
        hidden = h0
        hiddens = torch.zeros(max_length, 1, self.params.autoencoder.hidden_size, device=device)
        currToken = self.vocabulary.embeddings['<go>']
        softmax = torch.nn.Softmax()
        for index in range(max_length):
            currToken = currToken.unsqueeze(0).unsqueeze(0)
            out, hidden = self.generator(currToken, hidden)
            vocabLogits = self.hiddenToVocab(out[0, 0, :])

            # TODO add dropout
            vocabProbs = softmax(vocabLogits / self.params.temperature)
            if soft:
                currToken = torch.matmul(
                    vocabProbs, self.vocabulary.embeddings.weight)
            else:
                _, argmax = vocabProbs.max(1)
                currToken = self.vocabulary.embeddings[argmax]

        hiddens = torch.cat((h0, hiddens), dim=0)
        return hiddens

    def reconstructionLoss(self, outputs, targets):
        return torch.nn.functional.cross_entropy(outputs, targets)

    def adversarialLoss(self, x_real, x_fake, label):
        discriminator = self.discriminators[label]
        d_real = discriminator(x_real)
        d_fake = discriminator(x_fake)
        label = torch.FloatTensor([label])

        loss_d = self.adv_loss_criterion(d_real, label) + \
            self.adv_loss_criterion(d_fake, 1 - label)
        loss_g = self.adv_loss_criterion(d_fake, label)
        return loss_d, loss_g

    def _zeroGradients(self):
        self.autoencoder_optimizer.zero_grad()
        self.discriminator0_optimizer.zero_grad()
        self.discriminator1_optimizer.zero_grad()

    def _runSentence(self, encoder_input, generator_input, label, target):
        # auto-encoder
        # initialize the first hidden state of the encoder
        initialHidden = self.labelsTransform(label)
        initialHidden = initialHidden.unsqueeze(0).unsqueeze(0)
        initialHidden = torch.cat(
            (initialHidden, torch.zeros(1, 1, self.params.dim_z)), dim=2)

        # encode tokens and extract only content=hidden[:,:,dim_y:]
        content = self._encodeTokens(encoder_input, initialHidden)

        # generating the hidden states (yp, zp)
        originalHidden = self.labelsTransform(label)
        originalHidden = originalHidden.unsqueeze(0).unsqueeze(0)
        originalHidden = torch.cat(
            (originalHidden, content), dim=2)

        # generating the hidden states with inverted labels (yq, zp)
        transformedHidden = self.labelsTransform(1 - label)
        transformedHidden = transformedHidden.unsqueeze(0).unsqueeze(0)
        transformedHidden = torch.cat(
            (transformedHidden, content), dim=2)

        # reconstruction loss
        generatorOutput, h_teacher = self._generateTokens(
            generator_input, originalHidden)
        self.losses['reconstruction'] += self.reconstructionLoss(
            generatorOutput, target)

        # adversarial losses
        h_professor = self._generateWithPrevOutput(
            transformedHidden, self.params.max_length, soft=True)
        d_loss, g_loss = self.adversarialLoss(h_teacher, h_professor, label)
        self.losses['discriminator{0}'.format(label)] += d_loss
        self.losses['generator'] += g_loss

    def _sentencesToInputs(self, sentences):
        # transform sentences into embeddings
        encoder_inputs, generator_inputs, targets = \
            preprocessSentences(sentences)
        encoder_inputs = list(map(
            self.vocabulary.getEmbedding, encoder_inputs))
        generator_inputs = list(map(
            self.vocabulary.getEmbedding, generator_inputs))
        targets = list(map(
            self.vocabulary.getWordId, sentences))

        return encoder_inputs, generator_inputs, targets

    def _computeLosses(self, encoder_inputs, generator_inputs, targets, labels):
        self.losses = defaultdict(float)

        for index in range(len(encoder_inputs)):
            label = labels[index]
            target = targets[index]
            encoder_input = encoder_inputs[index]
            generator_input = generator_inputs[index]
            self._runSentence(encoder_input, generator_input, label, target)

    def trainOnBatch(self, sentences, labels):
        self.train()
        labels = np.array(labels)
        encoder_inputs, generator_inputs, targets = \
            self._sentencesToInputs(sentences)

        self._zeroGradients()
        self._computeLosses(encoder_inputs, generator_inputs, targets)

        self.losses['autoencoder'] = self.losses['reconstruction'] + \
            self.params.lambda_GAN * self.losses['generator']
        self.losses['autoencoder'] /= len(sentences)

        self.losses['autoencoder'].backward()
        self.autoencoder_optimizer.step()
        self._zeroGradients()

        self.losses['discriminator0'] /= len(sentences)
        self.losses['discriminator0'].backward()
        self.discriminator0_optimizer.step()
        self._zeroGradients()

        self.losses['discriminator1'] /= len(sentences)
        self.losses['discriminator1'].backward()
        self.discriminator1_optimizer.step()
        self._zeroGradients()
        return self.losses['autoencoder']

    def evaluate(self, sentences, labels):
        self.eval()
        self.losses = defaultdict(float)
        encoder_inputs, generator_inputs, targets = \
            self._sentencesToInputs(sentences)

        self._computeLosses(encoder_inputs, generator_inputs, targets, labels)

        self.losses['autoencoder'] = self.losses['reconstruction'] + \
            self.params.lambda_GAN * self.losses['generator']
        self.losses['autoencoder'] /= len(sentences)

        self.losses['discriminator0'] /= len(sentences)
        self.losses['discriminator1'] /= len(sentences)

        return self.losses['autoencoder']
