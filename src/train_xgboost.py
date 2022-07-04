import logging
import os
import pickle
import random

import numpy as np
import pandas as pd
import yaml

from SKLearnDS import SKlearnDS

import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['xgboost']

    # set random state
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Load the data
    logging.info('Loading data...')
    data_train = pd.read_csv("./data/train/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_train = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    unique_labels = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    # using config You can adjust how many random samples are used in training
    if config['num_training_samples'] > 0:
        # temporary combine train and labels
        combined = list(zip(data_train[0], labels_train[0]))
        # combined = sorted(combined, key=lambda x: len(x[1].split(" ")), reverse=True)

        # prepare "table" for labels where columns are labels and samples are rows, each sample can have multiple labels
        category_id = dict((label, set()) for label in unique_labels[0].unique())

        # fill the table
        for i, (doc, doc_labels) in enumerate(combined):
            single_doc_labels = doc_labels.split(" ")
            for label in single_doc_labels:
                category_id[label].add((i, len(single_doc_labels)))

        # sort each column by the number of labels in each sample
        for label in category_id.keys():
            category_id[label] = sorted(category_id[label], key=lambda x: x[1], reverse=True)

        # select samples for training
        items_ids = []
        used_ids = set()

        curr_id = 0
        while len(items_ids) < config['num_training_samples']:
            for label in category_id.keys():
                # print(label)
                if len(items_ids) >= config['num_training_samples']:
                    break

                try:
                    doc_id = category_id[label][curr_id][0]
                    if doc_id in used_ids:
                        pass

                    items_ids.append(doc_id)
                    used_ids.add(doc_id)
                except:
                    continue
            curr_id += 1

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
    boost = OneVsRestClassifier(xgb.XGBClassifier(objective='binary:logistic')).fit(ds_texts, ds_labels)

    # Save the model
    logging.info('Saving the model...')
    os.makedirs("./models/xgboost", exist_ok=True)
    with open("./models/xgboost/model.pkl", "wb") as f:
        pickle.dump(boost, f)

    with open("./models/xgboost/vectorizer.pkl", "wb") as f:
        pickle.dump(ds_train.count_vect, f)

    with open("./models/xgboost/labels.pkl", "wb") as f:
        pickle.dump(unique_labels, f)

    logging.info('Done!')
