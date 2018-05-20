"""
Keep the current vocabulary and embeddings
"""
import logging
import pickle
import torch


class Vocabulary(object):
    _SPECIAL_TOKENS = ['<pad>', '<go>', '<eos>', '<unk>']

    def __init__(self):
        "vocabulary is a list of all the words we are interested into"
        self.embeddings = None
        self.vocabulary = None

    def loadVocabulary(self, fileName):
        with open(fileName, 'rb') as fp:
            vocabulary = pickle.load(fp)

        self.vocabulary = vocabulary
        self.word2id = dict(zip(
            self._SPECIAL_TOKENS, range(len(self._SPECIAL_TOKENS))))
        self.id2word = self._SPECIAL_TOKENS
        for wordId, word in enumerate(self.vocabulary):
            currId = wordId + len(self._SPECIAL_TOKENS)
            self.word2id[word] = currId
            self.id2word.append(word)
        self.vocabSize = len(self.id2word)

    def initializeEmbeddings(self, embeddingSize):
        self.embeddingSize = embeddingSize
        if self.vocabulary is None:
            logging.error('Load vocabulary first')
            return

        self.embeddings = torch.nn.Embedding(
            self.vocabSize + 1, self.embeddingSize)

    def getSentenceIds(self, words):
        unkId = self.word2id['<unk>']
        ids = list(map(lambda x: self.word2id.get(x, unkId), words))
        return torch.LongTensor(ids)

    def getEmbedding(self, words):
        ids = self.getSentenceIds(words)
        return self.embeddings(ids)
