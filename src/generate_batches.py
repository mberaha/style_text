from sklearn.utils import shuffle


def batchesFromFiles(files: list, batchsize: int, inMemory: bool):
    if inMemory:
        return loadFilesAndGenerateBatches(files, batchsize)

    return yeildBatchesFromFiles(files, batchsize)


def yeildBatchesFromFiles(files, batchsize):
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


def loadFilesAndGenerateBatches(files, batchsize, shuffleFiles=True):
    inputs = []
    labels = []
    for label, fileName in enumerate(files):
        with open(fileName, 'r') as fp:
            lines = fp.readlines()

        labels.extend([label] * len(lines))
        inputs.extend(lines)

    if shuffleFiles:
        inputs, labels = shuffle(inputs, labels)

    batches = []
    for index in range(batchsize, len(inputs), batchsize):
        batches.append(
            (inputs[index:index+batchsize],
             labels[index:index+batchsize]))

    return batches
