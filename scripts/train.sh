#!/bin/bash

set -e

python src/train_wrapper.py "$@"
python src/rewrite_dataset.py
python src/train_roberta_lm.py
python src/train_roberta_classifier.py

mv data/dev/out.tsv ./dev-0/out.tsv
mv data/test/out.tsv ./test-A/out.tsv
rm -r ./data