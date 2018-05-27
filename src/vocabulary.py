"""
Keep the current vocabulary and embeddings
"""
import logging
import pickle
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SPECIAL_TOKENS = ['<pad>', '<go>', '<eos>', '<unk>']


class Vocabulary(object):

    def __init__(self):
        "vocabulary is a list of all the words we are interested into"
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
        print(self.vocabSize, max(list(self.word2id.values())))

    def initializeEmbeddings(self, embeddingSize):
        self.embeddingSize = embeddingSize
        if self.word2id is None:
            logging.error('Load vocabulary first')
            return

        self.embeddings = torch.nn.Embedding(
            self.vocabSize + 1, self.embeddingSize)

    def getSentenceIds(self, words):
        unkId = self.word2id['<unk>']
        ids = list(map(lambda x: self.word2id.get(x, unkId), words))
        return torch.LongTensor(ids, device=device)

    def getEmbedding(self, words):
        ids = self.getSentenceIds(words)
        return self.embeddings(ids)
