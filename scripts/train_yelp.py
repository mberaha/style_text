import argparse
import easydict
import glob
import logging
import torch
from src.generate_batches import batchesFromFiles
from src.parameters import Params
from src.style_transfer import StyleTransfer
from src.vocabulary import Vocabulary

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # unCOMMENT this if running from terminal
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train_files", type=str)
    # parser.add_argument("--evaluation_files", type=str)
    # parser.add_argument("--vocabulary", type=str)
    # args = parser.parse_args()

    # unCOMMENT this if running from environments like Notebooks, Hydrogen...
    args = easydict.EasyDict({
            "train_files": "data/yelp/train/*",
            "evaluation_files": "data/yelp/dev/*",
            "vocabulary": "data/yelp/vocabulary.pickle"
    })

    params = Params()
    vocab = Vocabulary()
    vocab.loadVocabulary(args.vocabulary)
    vocab.initializeEmbeddings(params.embedding_size)

    logging.info("beginning train_yelp")
    model = StyleTransfer(params, vocab)
    if torch.cuda.is_available():
        model = model.cuda()
    trainFiles = glob.glob(args.train_files)
    trainBatches = batchesFromFiles(trainFiles, params.batch_size, True)
    validFiles = glob.glob(args.evaluation_files)
    validSet = batchesFromFiles(validFiles, params.batch_size, inMemory=True)
    model.trainModel(trainBatches, validSet)
