import argparse
import easydict
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
            # "train_file_positive": "data/yelp/train/positive_sentence.txt",
            # "train_file_negative": "data/yelp/train/negative_sentence.txt",
            # "evaluation_file_positive": "data/yelp/dev/positive.txt",
            # "evaluation_file_negative": "data/yelp/dev/negative.txt",
            "train_file_positive": "data/yelp/test/positive_sentence.txt",
            "train_file_negative": "data/yelp/test/negative_sentence.txt",
            "evaluation_file_positive": "data/yelp/test/positive_sentence.txt",
            "evaluation_file_negative": "data/yelp/test/negative_sentence.txt",
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

    trainBatches = batchesFromFiles(
        positiveFile=args.train_file_positive,
        negativeFile=args.train_file_negative,
        batchsize=params.batch_size,
        inMemory=True)

    validSet = batchesFromFiles(
        positiveFile=args.evaluation_file_positive,
        negativeFile=args.evaluation_file_negative,
        batchsize=params.batch_size,
        inMemory=True)

    # print(trainBatches)
    # print("len(trainBatches)", len(trainBatches))
    # print("len(validSet)", len(validSet))

    model.trainModel(trainBatches, validSet)
