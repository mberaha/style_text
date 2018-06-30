import argparse
import logging
import torch
from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
from src.generate_batches import batchesFromFiles
from src.parameters_pb2 import StyleTransferParams
from src.style_transfer import StyleTransfer
from src.vocabulary import Vocabulary

machineToParams = {
    'local': 'resources/local_params.asciipb',
    'server': 'resources/server_params.asciipb'
}


def loadParams(machine):
    filename = machineToParams[machine]
    params = StyleTransferParams()
    with open(filename, 'r') as fp:
        text_format.Parse(fp.read(), params)
    return params


def printParams(params):
    print('Parameters:')
    print(MessageToJson(
        params,
        including_default_value_fields=True,
        preserving_proto_field_name=True))


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
    parser.add_argument("--machine", type=str, default="local")
    args = parser.parse_args()

    params = loadParams(args.machine)
    params.savefile = args.savefile
    params.logdir = args.logdir
    printParams(params)
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

    model.trainModel(trainBatches, validSet)
