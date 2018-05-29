"""
Get the dictionary word -> id for yelp datasets
Most common words get lower indices for memory efficiency
"""
import argparse
import glob
import pickle
import re
from collections import Counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", type=str)
    parser.add_argument("--vocabulary_output", type=str)
    args = parser.parse_args()

    vocabulary = Counter()
    fileNames = glob.glob(args.files, recursive=True)
    for fileName in fileNames:
        with open(fileName, 'r') as fp:
            text = fp.read()
        text = text.strip()
        text = re.sub("\n", " ", text)
        vocabulary.update(text.split(" "))

    wordsAndCounts = sorted(
        list(vocabulary.items()), key=lambda x: x[1], reverse=True)
    vocabulary = list(map(lambda x: x[0], wordsAndCounts))
    with open(args.vocabulary_output, 'wb') as fp:
        pickle.dump(vocabulary, fp)
