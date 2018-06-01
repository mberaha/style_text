import random
import torch
from torch import nn
from tqdm import tqdm


class BaseModel(nn.Module):
    _MAX_LOSS = 1e10

    def __init__(self):
        super().__init__()

    def load(self):
        self.load_state_dict(self.params.savefile)

    def trainModel(self, trainBatches, validBatch, shuffle=True):
        for epoch in range(self.params.epochs):
            if shuffle:
                random.shuffle(trainBatches)
            self.runEpoch(trainBatches, validBatch, epoch)

    def runEpoch(self, trainBatches, validBatch, epoch):
        # TODO risolvere visualizzazione doppia progbar
        bestLoss = self._MAX_LOSS
        progbar = tqdm(range(len(trainBatches)))
        for index in progbar:
            inputs, labels = trainBatches[index]
            loss = self.trainOnBatch(inputs, labels)
            progbar.set_description("Loss: {0}".format(loss))

        sentences = validBatch[0]
        labels = validBatch[1]
        print("batchsize in runEpoch:", len(sentences))
        evaluationLoss = self.evaluate(sentences, labels)
        print("Epoch {0}/{1}, Loss on evaluation set: {2}".format(
            epoch, self.params.epochs, evaluationLoss))
        if evaluationLoss < bestLoss:
            bestLoss = evaluationLoss
            torch.save(self.state_dict(), self.params.savefile)
