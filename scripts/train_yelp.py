import argparse
import glob
from src.generate_batches import batchesFromFiles
from src.parameters import Params
from src.style_transfer import StyleTransfer
from src.vocabulary import Vocabulary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_files", type=str)
    parser.add_argument("--evaluation_files", type=str)
    parser.add_argument("--vocabulary", type=str)
    args = parser.parse_args()

    params = Params()
    vocab = Vocabulary()
    vocab.loadVocabulary(args.vocabulary)
    vocab.initializeEmbeddings(params.embedding_size)

    model = StyleTransfer(params, vocab)
    trainFiles = glob.glob(args.train_files)
    trainBatches = batchesFromFiles(trainFiles, params.batch_size, True)
    validFiles = glob.glob(args.evaluation_files)
    validSet = batchesFromFiles(validFiles, -1, inMemory=True)
    model.trainModel(trainBatches, validSet)
