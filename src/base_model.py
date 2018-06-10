import datetime
import random
import torch
from torch import nn
from tqdm import tqdm


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.iter = 0

    def load(self):
        self.load_state_dict(self.params.savefile)

    def trainModel(self, trainBatches, validBatches, shuffle=True):
        for epoch in range(self.params.epochs):
            if shuffle:
                random.shuffle(trainBatches)
            self.runEpoch(trainBatches, validBatches, epoch)

    def runEpoch(self, trainBatches, validBatches, epoch):
        # TODO risolvere visualizzazione doppia progbar
        progbar = tqdm(range(len(trainBatches)))
        for index in progbar:
            self.iter += 1
            inputs, labels = trainBatches[index]
            loss = self.trainOnBatch(inputs, labels, self.iter)
            progbar.set_description("Loss: {0:.6f}".format(loss))

        evaluationLoss = self.evaluate(validBatches)
        tqdm.write("Epoch {0}/{1}, Loss on evaluation set: {2}".format(
            epoch + 1, self.params.epochs, evaluationLoss))
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        fileName = '{0}-{1}-epoch_{2}-loss_{3}'.format(
            self.params.savefile, date, index, "{0:4f}".format(evaluationLoss))
        torch.save(self.state_dict(), fileName)
