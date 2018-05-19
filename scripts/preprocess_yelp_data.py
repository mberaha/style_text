import argparse
import glob
import pickle
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str)
    parser.add_argument("--vocabulary_output", type=str)
    args = parser.parse_args()

    vocabulary = Counter()
    fileNames = glob.glob(args.files, recursive=True)
    print(fileNames)
    for fileName in fileNames:
        with open(fileName, 'r') as fp:
            text = fp.read()

        vocabulary.update(text.split(" "))

    wordsAndCounts = sorted(
        list(vocabulary.items()), key=lambda x: x[1], reverse=True)
    wordsByCount = list(map(lambda x: x[0], wordsAndCounts))
    vocabulary = dict(zip(wordsByCount, range(len(wordsByCount))))
    with open(args.vocabulary_output, 'wb') as fp:
        pickle.dump(vocabulary, fp)
