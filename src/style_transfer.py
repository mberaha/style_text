import numpy as np
import torch
from torch import optim
import torch.nn as nn
from collections import defaultdict
from src.generate_batches import preprocessSentences
from src.rnn import Rnn
from src.discriminator import Cnn
from src.vocabulary import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StyleTransfer(object):

    def __init__(self, params, vocabulary: Vocabulary):

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
            params.hidden_size, self.vocabulary.vocabSize + 1)

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
            self.discriminator.parameters(),
            lr=params.discriminator.learning_rate,
            betas=params.discriminator.betas)
        self.discriminator1_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=params.discriminator.learning_rate,
            betas=params.discriminator.betas)

        # instantiating the loss criterion
        self.rec_loss_criterion = nn.CrossEntropyLoss()
        self.adv_loss_criterion = nn.BCEWithLogitsLoss()


        # instantiating some useful functions

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

    def _generateWithPrevOutput(self, h0, length, soft=True):
        hidden = h0
        hiddens = torch.zeros(length, 1, self.params.hidden_size, device=device)
        currToken = self.vocabulary.embeddings['<go>']
        softmax = torch.nn.Softmax()
        for index in range(length):
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

    def adversarialLoss(x_real, x_fake, label):
        discriminator = self.discriminators[label]
        d_real = self.discriminator(x_real)
        d_fake = self.discriminator(x_fake)
        label = torch.FloatTensor([label])

        loss_d = self.adv_loss_criterion(d_real, label) + \
            self.adv_loss_criterion(d_fake, 1 - label)
        loss_g = self.adv_loss_criterion(d_fake, label)
        return loss_d, loss_g

    def trainOnBatch(self, sentences, labels):
        # transform sentences into embeddings
        labels = np.array(labels)
        encoder_inputs, decoder_inputs, targets = preprocessSentences(sentences)
        encoder_inputs = list(map(self.vocabulary.getEmbedding, encoder_inputs))
        decoder_inputs = list(map(self.vocabulary.getEmbedding, decoder_inputs))
        targets = list(map(self.vocabulary.getWordId, sentences))

        self.losses = defaultdict(float)
        self.encoder_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()
        self.discriminator0_optimizer.zero_grad()
        self.discriminator1_optimizer.zero_grad()
        originalHidden = []
        transformedHidden = []
        for index, sentence in enumerate(sentences):

            #####   auto-encoder   #####
            # initialize the first hidden state of the encoder
            initialHidden = self.labelsTransform(labels[index])
            initialHidden = initialHidden.unsqueeze(0).unsqueeze(0)
            initialHidden = torch.cat(
                (initialHidden, torch.zeros(1, 1, self.params.dim_z)), dim=2)

            # encode tokens and extract only content=hidden[:,:,dim_y:]
            content = self._encodeTokens(sentence, initialHidden)


            # generating the hidden states (yp, zp)
            originalHidden = self.labelsTransform(labels[index])
            originalHidden = originalHidden.unsqueeze(0).unsqueeze(0)
            originalHidden = torch.cat(
                (originalHidden, content), dim=2)

            # generating the hidden states with inverted labels (yq, zp)
            transformedHidden = self.labelsTransform(1 - labels[index])
            transformedHidden = transformedHidden.unsqueeze(0).unsqueeze(0)
            transformedHidden = torch.cat(
                (transformedHidden, content), dim=2)

            self.generator_optimizer.zero_grad()

            # reconstruction loss
            generatorOutput, h_teacher = self._generateTokens(sentence, originalHidden)
            self.losses['reconstruction'] += self.reconstructionLoss(
                generatorOutput, targets)

            # adversarial losses
            h_professor = self._generateWithPrevOutput(
                transformedHidden, len(sentence), soft=True)
            d_loss, g_loss = self.adversarialLoss(h_teacher, h_professor, label)
            self.losses['discriminator{0}'.format(label)] += d_loss
            self.losses['generator'] += g_loss

        loss_autoencoder = self.losses['reconstruction'] + \
            self.params.lambda_GAN * self.losses['generator']
        loss_autoencoder /= len(sentences)

        loss_autoencoder.backward()
        self.autoencoder_optimizer.step()

        self.losses['']
        self.discriminator0_optimizer.step()
        self.discriminator1_optimizer.step()
