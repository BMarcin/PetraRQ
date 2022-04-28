import logging
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
from transformers import TextClassificationPipeline, RobertaForSequenceClassification, RobertaTokenizerFast

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    #todo adjust

    # load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['classification_train']
    config_train = yaml.safe_load(open("./params.yaml"))['language_modeling_train']
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in config['cuda_visible_devices']])

    # set random state
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # setup datasets paths
    dev_ds = "./data/dev/"
    test_ds = "./data/test/"

    lm_model_path = "./models/roberta_lm"
    models_path = "./models/roberta_classifier"

    # define special characters
    logging.info("Defining special characters...")
    special_tokens = [
        '<url>',
        '<email>',
        '<number>',
        '<date>',
    ]

    logging.info("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        lm_model_path,
        use_fast=True)

    # Load the data
    logging.info('Loading data...')
    data_test= pd.read_csv("./data/test/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data_dev = pd.read_csv("./data/dev/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_test = pd.read_csv("./data/test/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_dev = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    # Make unique labels
    logging.info('Making unique labels...')
    unique_labels_tsv = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    unique_labels = unique_labels_tsv[0].tolist()

    # define model
    logging.info("Defining model...")
    # model = RobertaForSequenceClassification.from_pretrained(models_path)
    pipe = TextClassificationPipeline(
        model=RobertaForSequenceClassification.from_pretrained(models_path, ),
        tokenizer=tokenizer,
        return_all_scores=True,
        max_length=config_train['max_seq_length'],
        truncation=True,
    )

    # predict dev set
    logging.info("Predicting dev set...")
    dev_probabilities = []
    dev_predictions = pipe(list(data_dev[0]), batch_size=config['per_device_eval_batch_size'])

    # rewrite predictions
    logging.info("Rewriting predictions...")
    for probe in tqdm(dev_predictions, desc="Predicting dev set"):
        labels_probabilities = []
        for label, label_name in zip(probe, unique_labels):
            labels_probabilities.append("{}:{:.9f}".format(label_name, label['score']))
        dev_probabilities.append(" ".join(labels_probabilities))

    # write predictions to file
    logging.info("Writing predictions to file...")
    with open("./data/dev/out.tsv", "w") as f:
        for line in dev_probabilities:
            f.write(line + "\n")

    # predict test set
    logging.info("Predicting test set...")
    test_probabilities = []
    test_predictions = pipe(list(data_test[0]), batch_size=config['per_device_eval_batch_size'])

    # rewrite predictions
    logging.info("Rewriting predictions...")
    for probe in tqdm(test_predictions, desc="Predicting test set"):
        labels_probabilities = []
        for label, label_name in zip(probe, unique_labels):
            labels_probabilities.append("{}:{:.9f}".format(label_name, label['score']))
        test_probabilities.append(" ".join(labels_probabilities))

    # write predictions to file
    logging.info("Writing predictions to file...")
    with open("./data/test/out.tsv", "w") as f:
        for line in test_probabilities:
            f.write(line + "\n")
