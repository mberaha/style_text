"""
Keep the current vocabulary and embeddings
"""
from torch.nn import Embedding


class Vocabulary(object):
    def __init__(self):
        self.embeddings = None

    def loadGloveEmbeddings(fileName):
