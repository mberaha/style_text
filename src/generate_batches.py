import copy
from sklearn.utils import shuffle
import numpy as np


def batchesFromFiles(style1, style2, batchsize, inMemory):
    if inMemory:
        return loadFilesAndGenerateBatches(
            style1, style2, batchsize)

    return yieldBatchesFromFiles(style1, style2, batchsize)


def yieldBatchesFromFiles(files, batchsize):
    """
    Generate batches without loading files in memory
    """
    openedFiles = []
    for fname in files:
        openedFiles.append(open(fname, 'r'))

    while True:
        inputs = []
        labels = []
        for label, fp in enumerate(openedFiles):
            for i in range(batchsize // len(files)):
                # remove final '\n'
                inputs.append(fp.readline()[:-1])
                labels.append(label)

        yield inputs, labels


def loadFilesAndGenerateBatches(
        style1, style2, batchsize=-1, shuffleFiles=True):
    inputs = []
    lenLines = []
    for label, fileName in enumerate([style1, style2]):
        with open(fileName, 'r') as fp:
            lines = fp.readlines()

        lines = list(map(lambda x: x[:-1], lines))
        # the last line is always an empty line
        lines = lines[:-1]
        lenLines.append(len(lines))
        if shuffleFiles:
            lines = shuffle(lines)

        inputs.append(lines)

    if batchsize < 0:
        labels = []
        sentences = []
        for index, examples in enumerate(inputs):
            sentences.extend(examples)
            labels.extend([index] * len(examples))
        return sentences, labels

    batches = []
    iterStep = batchsize // len(inputs)
    for index in range(0, min(lenLines), iterStep):
        currInputs = []
        currLabels = []
        currInputs.extend(inputs[0][index:index + iterStep])
        currLabels.extend([0] * iterStep)
        currInputs.extend(inputs[1][index:index + iterStep])
        currLabels.extend([1] * iterStep)
        if len(currInputs) == batchsize:
            batches.append((currInputs, currLabels))
    return batches


def noise(sentences, word_drop=0.0, k=3):
    """
    Apply noise to input sentences as suggested in the paper:
    Unsupervised Machine Translation Using Monolingual Corpora Only
    """
    unk = '<unk>'
    for sentIndex, sent in enumerate(sentences):
        sentLen = len(sent)
        for wordIndex in range(sentLen):
            # drop words from input with probability word-drop
            if np.random.random_sample() < word_drop:
                sent[wordIndex] = unk

        # slightly shuffle the input sequences by applying a random
        # permutation sigma  which verifies the condition:
        # |sigma[word]-word| <= k
        noisyIndexes = np.arange(sentLen) + (k+1) * np.random.rand(sentLen)
        sigma = (noisyIndexes).argsort()
        sent = [sent[sigma[word]] for word in range(sentLen)]
        sentences[sentIndex] = sent

    return sentences


def preprocessSentences(
        sentences, padToMaxLen=True, noisy=False, word_drop=0.0):
    def addGo(sentence):
        out = ['<go>']
        out.extend(sentence)
        return out

    def addEos(sentence):
        sentence.append('<eos>')
        return sentence

    def addPad(sentence, maxLen):
        currLen = len(sentence)
        sentence.extend(['<pad>'] * (maxLen - currLen))
        return sentence

    sentences = sorted(sentences, key=len, reverse=True)
    if noisy:
        sentences = noise(sentences, word_drop)
    encoder_inputs = copy.deepcopy(sentences)
    encoder_inputs = [addEos(x) for x in encoder_inputs]
    decoder_inputs = copy.deepcopy(sentences)
    decoder_inputs = [addGo(x) for x in decoder_inputs]
    lengths = list(map(len, encoder_inputs))
    targets = copy.deepcopy(encoder_inputs)
    if padToMaxLen:
        maxlen = len(encoder_inputs[0])
        encoder_inputs = [addPad(x, maxlen) for x in encoder_inputs]
        decoder_inputs = [addPad(x, maxlen) for x in decoder_inputs]
        targets = [addPad(x, maxlen) for x in targets]

    return encoder_inputs, decoder_inputs, targets, lengths
