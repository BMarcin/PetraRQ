import json
import logging
import os
import pickle
import random

import pandas as pd
import yaml

import fasttext
import torch
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def labels2tensor(labels, label2idx):
    # print(labels)
    unique = set([label2idx[label.strip().lower()] for label in labels])
    if len(unique) == 0:
        return torch.zeros([len(label2idx)], dtype=torch.long).tolist()
    return torch.zeros([len(label2idx)]).index_fill_(0, torch.tensor(list(unique)), 1).tolist()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['fasttext']

    # set random state
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Load the data
    logging.info('Loading data...')
    data_train = pd.read_csv("./data/train/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_train = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    if config['num_training_samples'] > 0:
        items_ids = []

        total_items = list(range(len(data_train)))
        for i in range(config["num_training_samples"]):
            items_ids.append(random.choice(total_items))
            item_index = total_items.index(items_ids[-1])
            del total_items[item_index]

        data_train = data_train.loc[items_ids]
        labels_train = labels_train.loc[items_ids]

    # Save data for fasttext
    logging.info('Saving data for fasttext...')
    with open("./data/train/fasttext.txt", "w", encoding="utf8") as f:
        for inpt, labels in zip(data_train[0], labels_train[0]):
            new_labels = " ".join(["__label__" + label for label in labels.split(" ")])

            f.write(new_labels + " " + inpt + "\n")

        # Make unique labels
        logging.info('Making unique labels...')
        unique_labels_tsv = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
        unique_labels = unique_labels_tsv[0].tolist()

        label2idx = {}
        for label in unique_labels:
            label2idx[label] = len(label2idx)

    # Train the model
    logging.info('Training the model...')
    model = fasttext.train_supervised(input="./data/train/fasttext.txt", loss='ova', epoch=config['epochs'], lr=config['lr'])

    # Save the model
    logging.info('Saving the model...')
    os.makedirs("./models/fasttext", exist_ok=True)
    model.save_model("./models/fasttext/fasttext.bin")
