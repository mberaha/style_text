#!/bin/bash
python3 -m scripts.train_model \
  --train_file_style1 data/yelp/debug/positive_sentence.txt \
  --train_file_style2 data/yelp/debug/negative_sentence.txt \
  --evaluation_file_style1 data/yelp/debug/positive_sentence.txt \
  --evaluation_file_style2 data/yelp/debug/negative_sentence.txt \
  --vocabulary data/yelp/vocabulary.pickle \
  --savefile data/models/yelp/model \
  --logdir data/models/yelp/log/
