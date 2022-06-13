import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import yaml

from SKLearnDS import SKlearnDS

from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['random_forest']

    # set random state
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Load the data
    logging.info('Loading data...')
    data_train = pd.read_csv("./data/train/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_train = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    # using config You can adjust how many random samples are used in training
    if config['num_training_samples'] > 0:
        items_ids = []

        total_items = list(range(len(data_train)))
        for i in range(config["num_training_samples"]):
            items_ids.append(random.choice(total_items))
            item_index = total_items.index(items_ids[-1])
            del total_items[item_index]

        data_train = data_train.loc[items_ids]
        labels_train = labels_train.loc[items_ids]

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
    forest = OneVsRestClassifier(RandomForestClassifier(n_estimators=config['n_estimators'], max_depth=config['max_depth'])).fit(ds_texts, ds_labels)

    # Save the model
    logging.info('Saving the model...')
    os.makedirs("./models/random_forest", exist_ok=True)
    with open("./models/random_forest/model.pkl", "wb") as f:
        pickle.dump(forest, f)

    with open("./models/random_forest/vectorizer.pkl", "wb") as f:
        pickle.dump(ds_train.count_vect, f)

    with open("./models/random_forest/labels.pkl", "wb") as f:
        pickle.dump(unique_labels, f)

    logging.info('Done!')
