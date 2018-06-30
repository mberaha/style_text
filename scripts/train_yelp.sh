#!/bin/bash

python3 -m scripts.train \
  --train_file_style1 data/yelp/train/positive_sentence.txt \
  --train_file_style2 data/yelp/train/negative_sentence.txt \
  --evaluation_file_style1 data/yelp/dev/positive.txt \
  --evaluation_file_style2 data/yelp/dev/negative.txt \
  --savefile edata/models/yelp/model
