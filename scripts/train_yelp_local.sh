#!/bin/bash

MACHINE=${1}

if [ -z "${MACHINE}" ]; then
  MACHINE="local"
fi

python3 -m scripts.train_model \
  --train_file_style1 data/yelp/test/positive_sentence.txt \
  --train_file_style2 data/yelp/test/negative_sentence.txt \
  --evaluation_file_style1 data/test/positive_sentence.txt \
  --evaluation_file_style2 data/test/negative_sentence.txt \
  --vocabulary data/yelp/vocabulary.pickle \
  --savefile data/models/yelp/model \
  --logdir data/models/yelp/log/ \
  --machine ${MACHINE}
