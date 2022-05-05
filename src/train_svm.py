import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import yaml

from SKLearnDS import SKlearnDS

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['svm']

    # set random state
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Load the data
    logging.info('Loading data...')
    data_train = pd.read_csv("./data/train/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_train = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    # Make unique labels
    logging.info('Making unique labels...')
    unique_labels_tsv = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    unique_labels = unique_labels_tsv[0].tolist()

    label2idx = {}
    for label in unique_labels:
        label2idx[label] = len(label2idx)

    # Build datasets
    logging.info('Building datasets...')
    ds_train = SKlearnDS(data_train[0], labels_train, label2idx)
    ds_texts, ds_labels = ds_train.get()

    # Train the model
    logging.info('Training the model...')
    svc = OneVsRestClassifier(LinearSVC()).fit(ds_texts, ds_labels)

    # Save the model
    logging.info('Saving the model...')
    os.makedirs("./models/svm", exist_ok=True)
    with open("./models/svm/model.pkl", "wb") as f:
        pickle.dump(svc, f)

    with open("./models/svm/vectorizer.pkl", "wb") as f:
        pickle.dump(ds_train.count_vect, f)

    with open("./models/svm/labels.pkl", "wb") as f:
        pickle.dump(unique_labels, f)
