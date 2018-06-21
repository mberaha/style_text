from src.parameters import Params
from src.vocabulary import Vocabulary
from src.style_transfer import StyleTransfer
from src.beam_search import Decoder
import torch
import numpy as np

params = Params()
vocab = Vocabulary()
vocab.loadVocabulary("data/yelp/vocabulary.pickle")
vocab.initializeEmbeddings(params.embedding_size)
model = StyleTransfer(params, vocab)

# %%
checkpoint = torch.load(
    "/home/lupol/dev/azurevm/final-2018-06-13-epoch_17-loss_6.839664")
# model.load_state_dict(checkpoint)

with open('data/yelp/test/negative_test.txt', 'r') as fp:
    testSents = fp.readlines()
print(testSents)
# %%
labels = np.array([1] * len(testSents))
decoder = Decoder(model, 20, 12, params)
orig, tsf = decoder.rewriteBatch(testSents[:12], labels[:12])

# %%
print(orig)
print(len(orig))
for i in range(len(orig)):
    print(len(orig[i]))

# %%
print(tsf)
print(len(tsf))
print(len(tsf[0]))
print(len(tsf[1]))
