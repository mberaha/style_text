import copy
import itertools
from sklearn.utils import shuffle


def batchesFromFiles(files: list, batchsize: int, inMemory: bool):
    if inMemory:
        return loadFilesAndGenerateBatches(files, batchsize)

    return yieldBatchesFromFiles(files, batchsize)


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


def loadFilesAndGenerateBatches(files, batchsize=-1, shuffleFiles=True):
    inputs = []
    lenLines = []
    for label, fileName in enumerate(files):
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
    print("len iterstep: ", iterStep)
    for index in range(0, min(lenLines), iterStep):
        currInputs = []
        currLabels = []
        for label, class_inputs in enumerate(inputs):
            currInputs.extend(class_inputs[index:index + iterStep])
            currLabels.extend([label] * iterStep)
        print("pre if: ", len(currLabels))
        if len(currLabels) == batchsize:
            print("passa", len(currLabels))
            batches.append((currInputs, currLabels))
    return batches


def preprocessSentences(sentences, padToMaxLen=True):
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
