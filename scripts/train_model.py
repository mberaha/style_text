import argparse
import logging
import torch
from src.generate_batches import batchesFromFiles
from src.parameters import Params
from src.style_transfer import StyleTransfer
from src.vocabulary import Vocabulary

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_style1", type=str)
    parser.add_argument("--train_file_style2", type=str)
    parser.add_argument("--evaluation_file_style1", type=str)
    parser.add_argument("--evaluation_file_style2", type=str)
    parser.add_argument("--vocabulary", type=str)
    parser.add_argument("--savefile", type=str)
    parser.add_argument("--logdir", type=str, default="")
    args = parser.parse_args()

    params = Params()
    params.savefile = args.savefile
    params.logdir = args.logdir
    vocab = Vocabulary()
    vocab.loadVocabulary(args.vocabulary)
    vocab.initializeEmbeddings(params.embedding_size)

    logging.info("beginning train_yelp")
    model = StyleTransfer(params, vocab)
    if torch.cuda.is_available():
        model = model.cuda()

    trainBatches = batchesFromFiles(
        style1=args.train_file_style1,
        style2=args.train_file_style2,
        batchsize=params.batch_size,
        inMemory=True)

    validSet = batchesFromFiles(
        style1=args.evaluation_file_style1,
        style2=args.evaluation_file_style2,
        batchsize=params.batch_size,
        inMemory=True)

    model.trainModel(trainBatches[:10], validSet[:10])
