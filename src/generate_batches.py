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
        lenLines.append(len(lines))
        if shuffleFiles:
            lines = shuffle(lines)

        inputs.append(lines)

    if batchsize < 0:
        labels = []
        for index, examples in enumerate(inputs):
            labels.extend([index] * len(examples))
        inputs = itertools.chain(*inputs)
        return inputs, labels

    batches = []
    iterStep = batchsize // len(inputs)
    for index in range(0, min(lenLines), iterStep):
        currInputs = []
        currLabels = []
        for label, class_inputs in enumerate(inputs):
            currInputs.extend(class_inputs[index:index + iterStep])
            currLabels.extend([label] * iterStep)
        batches.append((currInputs, currLabels))
    return batches


def preprocessSentences(sentences):
    def addGo(sentence):
        out = ['<go>']
        out.extend(sentence)
        return out

    def addEos(sentence):
        sentence.append('<eos>')
        return sentence

    encoder_inputs = sentences
    decoder_inputs = [addGo(x) for x in sentences]
    targets = [addEos(x) for x in sentences]
    return encoder_inputs, decoder_inputs, targets
