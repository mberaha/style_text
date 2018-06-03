import torch
import torch.nn.functional as F
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BeamState(object):

    def __init__(self, word, h, sentence, nll):
        """
        Args:
            word -- the id of the word charachterising the state
            h -- the hidden state associated to that state
            sentence -- a list of word ids (the past ids plus the current one)
            nll -- the negative log likelihood corresponding to the sentence
        """
        self.word, self.h, self.sentence, self.nll = \
            input, h, sentence, nll


class Decoder(object):

    def __init__(self, rnn, hToVocab, vocabulary, max_length, beam_width):
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.width = beam_width
        self.generator = rnn
        self.hToVocab = hToVocab

    def _decode(self, tokens, h):
        """
        Args:
            tokens --
            h --
        Outputs:
            logProbs --
            indices --
            h --
        """
        # embedding nn lookup
        # currTokens = torch.matmul(
        #     vocabProbs, self.vocabulary.embeddings.weight)

        currTokens = tokens
        currh = h
        # generate next h state and logit
        # generator needs input (seq_len, batch_size, input_size)
        outs, h = self.generator(currTokens, currh, pad=False)
        vocabLogits = self.hToVocab(outs[:, 0, :])
        # smooth logits into the probabilities of each word
        vocabProbs = F.softmax(
            vocabLogits / self.params.temperature, dim=1)
        # beam search trick to prevent probs vanishing
        logProbs = torch.log(vocabProbs)
        # take the beam_with most probable words
        logProbs, indices = torch.topk(logProbs, self.width, dim=-1)

        return logProbs, indices, h

    def _beamDecode(self, h0):
        """
        Returning the ids of the beam_width most probable sentences' words.

        Args:
            h0 -- the first hidden state of dim = dim_y + dim_z
        """

        go = self.vocabulary.getEmbedding(['<go>']).squeeze(0)
        batch_size = len(h0)
        init_state = BeamState([go]*batch_size, h0,
                               [[] for i in range(batch_size)], [0]*batch_size)
        beam = torch.FloatTensor([init_state])
        beam.to(device)
        for _ in range(self.max_len):
            storeBeamLayer = torch.FloatTensor(
                [[] for _ in range(batch_size)])
            storeBeamLayer.to(device)
            for state in beam:
                logProbs, indices, h = self._decode(state.inp, state.h)
                for b in range(batch_size):
                    for w in range(self.beam_width):
                        storeBeamLayer[b].append(
                            BeamState(indices[b, w], h[b],
                                      state.sentence[b] + [indices[b, w]],
                                      state.nll[b] - logProbs[b, w]))

            beam = torch.FloatTensor(
                [deepcopy(init_state) for _ in range(self.beam_width)])
            beam.to(device)
            for b in range(batch_size):
                # sort beam states by their probability (cumulated nll)
                # TODO check if performance increase by dividing nll
                # by number of words
                sortedBeamLayer = sorted(storeBeamLayer[i], key=lambda k: k.nll)
                for w in range(self.beam_width):
                    beam[w].word[b] = sortedBeamLayer[w].word
                    beam[w].h[b] = sortedBeamLayer[w].h
                    beam[w].sentence[b] = sortedBeamLayer[w].sentence
                    beam[w].nll[b] = sortedBeamLayer[w].nll

        # Returning the ids of the beam_width most probable sentences' words.
        sentences = beam[0].sent
        sentences = \
            [[self.vocabulary.id2word[i] for i in sent] for sent in sentences]
        # TODO strip the EOS
        # sentences = strip_eos(sentences)

        return sentences
