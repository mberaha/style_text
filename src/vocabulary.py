"""
Keep the current vocabulary and embeddings
"""
import logging
import pickle
import torch
import numpy as np
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SPECIAL_TOKENS = ['<pad>', '<go>', '<eos>', '<unk>']


class Vocabulary(nn.Module):

    def __init__(self):
        "vocabulary is a list of all the words we are interested into"
        super().__init__()
        self.embeddings = None
        self.word2id = None

    def loadVocabulary(self, fileName):
        with open(fileName, 'rb') as fp:
            vocabulary = pickle.load(fp)

        self.word2id = dict(zip(
            _SPECIAL_TOKENS, range(len(_SPECIAL_TOKENS))))
        self.id2word = _SPECIAL_TOKENS
        numSpecial = len(_SPECIAL_TOKENS)
        for wordId, word in enumerate(vocabulary):
            currId = wordId + numSpecial
            self.word2id[word] = currId
            self.id2word.append(word)
        self.vocabSize = len(self.id2word)

    def initializeEmbeddings(self, embeddingSize):
        self.embeddingSize = embeddingSize
        if self.word2id is None:
            logging.error('Load vocabulary first')
            return

        self.embeddings = torch.nn.Embedding(
            self.vocabSize, self.embeddingSize).to(device)

    def noise(IDs, unk, word_drop=0.0, k=3):
        """
        Apply noise to input sentences as suggested in the paper:
        Unsupervised Machine Translation Using Monolingual Corpora Only
        """
        batchSize = IDs.shape[0]
        for sent in range(batchSize):
            n = len(sent)
            for i in range(n):
                if np.random.random_sample() < word_drop:
                    # enters here only if word_drop>0.0
                    sent[i] = unk

            # slight shuffle such that |sigma[i]-i| <= k
            sigma = (np.arange(n) + (k+1) * np.random.rand(n)).argsort()
        return [x[sigma[i]] for i in range(n)]

        return

    def getSentenceIds(self, words):
        unkId = self.word2id['<unk>']
        ids = list(map(lambda x: self.word2id.get(x, unkId), words))
        return torch.LongTensor(ids).to(device)

    def getEmbedding(self, words, byWord):
        if byWord:
            unkId = self.word2id['<unk>']
            wordsID = self.getSentenceIds(words)
            wordsID = noise(wordsID, unkID) if noisy else wordID
        return self.embeddings(wordsID)

    def forward(self, inputs, byWord=True):
        return self.getEmbedding(inputs, byWord)
