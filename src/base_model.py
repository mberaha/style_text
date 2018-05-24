import torch
from torch import nn
from tqdm import tqdm


class BaseModel(nn.Module):
    _MAX_LOSS = 1e10

    def __init__(self, savefile):
        super().__init__()
        self.savefile = savefile

    def load(self):
        self.load_state_dict(self.savefile)

    def train(self, trainBatches, validBatch):
        bestLoss = self._MAX_LOSS
        for index in tqdm(range(len(trainBatches))):
            inputs, labels = trainBatches[index]
            self.trainOnBatch(inputs, labels)

        evaluationLoss = self.evaluate(*validBatch)
        if evaluationLoss < bestLoss:
            torch.save(self.state_dict())
