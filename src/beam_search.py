import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional af F
from collections import defaultdict
from src.rnn import Rnn
from src.vocabulary import Vocabulary


class BeamState(object):
    def __init__(self, input, h, sentence, nll):
        self.inp, self.h, self.sent, self.nll = h, input, sentence, nll


class Decoder(object):

    def __init__(self, rnn, hiddenToVocab, vocabulary, max_length, beam_width):
        super().__init__()
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.width = beam_width
        self.generator = rnn
        self.hiddenToVocab = hiddenToVocab

    def _decode(tokens, hidden):

        # embedding nn lookup
        # currTokens = torch.matmul(
        #     vocabProbs, self.vocabulary.embeddings.weight)

        currTokens = tokens
        currHiddens = hidden
        # generate next hidden state and logit
        # generator needs input (seq_len, batch_size, input_size)
        out, hidden = self.generator(currTokens, currHidden, pad=False)
        vocabLogits = self.hiddenToVocab(out[:, 0, :])
        # smooth logits into the probabilities of each word
        vocabProbs = F.softmax(
            vocabLogits / self.params.temperature, dim=1)
        # beam search trick to prevent probs vanishing
        logProbs = torch.log(vocabProbs)
        # take the beam_with most probable words
        logProbs, indices = torch.topk(logProbs, self.width, dim=-1)

    def _beamDecode(self, h0, lengths):

        go = self.vocabulary.getEmbedding(['<go>']).squeeze(0)
        batch_size = len(h0)
        init_state = BeamState(h0, [go]*batch_size,
                               [[] for i in range(batch_size)], [0]*batch_size)
        beam = [init_state]
        for t in range(self.max_len):
            exp = [[] for i in range(batch_size)]
            for state in beam:
                log_lh, indices, h = _decode(state.inp, state.h)
                for i in range(batch_size):
                    for l in range(self.beam_width):
                        exp[i].append(
                            BeamState(h[i], indices[i,l],
                            state.sent[i] + [indices[i,l]],
                            state.nll[i] - log_lh[i,l]))

            beam = [deepcopy(init_state) for _ in range(self.beam_width)]
            for i in range(batch_size):
                a = sorted(exp[i], key=lambda k: k.nll)
                for k in range(self.beam_width):
                    beam[k].h[i] = a[k].h
                    beam[k].inp[i] = a[k].inp
                    beam[k].sent[i] = a[k].sent
                    beam[k].nll[i] = a[k].nll
        return beam[0].sent

    def rewrite(self, h0, lengths):

        # decode sentences of word-ids through beam search
        sentences = self._beamDecode(h0, lengths)
        # convert word-ids to actual words
        sentences = \
            [[self.vocabulary.id2word[i] for i in sent] for sent in sentences]
        # TODO strip the EOS
        # sentences = strip_eos(sentences)

        return sentences
