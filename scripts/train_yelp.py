import argparse
from src.generate_batches import batchesFromFiles
from src.parameters import Params
from src.style_transfer import StyleTransfer
from src.vocabulary import Vocabulary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_files", type=str)
    parser.add_argument("--evaluation_files", type=str)
    parser.add_argumetn("--vocabulary", type=str)
    args = parser.parse_args()

    params = Params()
    vocab = Vocabulary
    vocab.loadVocabulary(args.vocabylary)
    vocab.initializeEmbeddings(params.embedding_size)

    model = StyleTransfer(params, vocab)
    trainBatches = batchesFromFiles(args.train_files, params.batchsize)
    validSet = batchesFromFiles(args.evaluation_files)
    model.train(trainBatches, validSet)
