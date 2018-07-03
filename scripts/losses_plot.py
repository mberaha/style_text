import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict


def getLossesFromFile(base_path):
    i = 0
    losses = {}
    while True:
        fileName = base_path.format(i)
        try:
            with open(fileName, 'rb') as fp:
                losses[i] = pickle.load(fp)
            i += 1
        except Exception as e:
            return losses, i


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path", type=str,
        default='data/models/yelp/log/epoch_{0}/losses.pickle')
    parser.add_argument(
        "--output", type=str)

    args = parser.parse_args()

    losses, n_epochs = getLossesFromFile(args.log_path)
    avgLossesById = defaultdict(list)
    stdDevById = defaultdict(list)
    for label in ['reconstruction', 'generator', 'autoencoder']:
        for i, epoch in losses.items():
            loss = []
            for batch in epoch:
                loss.append(float(batch[label]))
            avgLossesById[label].append(np.average(loss))
            stdDevById[label].append(np.std(loss))

    for i, epoch in losses.items():
        loss = []
        for batch in epoch:
            loss.append(batch['discriminator0'] + batch['discriminator1'])
        avgLossesById['discriminators'].append(np.average(loss))
        stdDevById['discriminators'].append(np.std(loss))

    labels = ['reconstruction', 'generator', 'autoencoder', 'discriminators']
    for label in labels:
        plt.errorbar(
            np.array(range(n_epochs)) + 1,
            avgLossesById[label],
            2 * np.array(stdDevById[label]))

    plt.legend(labels, loc="best")
    plt.savefig(args.output)
