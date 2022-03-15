import logging
import os

import pandas as pd
import torch
import yaml
from transformers import TextClassificationPipeline, RobertaForSequenceClassification, RobertaTokenizerFast

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['classification_train']
    config_train = yaml.safe_load(open("./params.yaml"))['language_modeling_train']
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in config['cuda_visible_devices']])

    # setup datasets paths
    dev_ds = "./data/dev/"
    test_ds = "./data/test/"
    train_ds = "./data/train/"

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
    tokenizer = RobertaTokenizerFast.from_pretrained(lm_model_path, max_len=config_train['max_seq_length'],
                                                     use_fast=True)

    # load datasets
    logging.info("Loading datasets...")
    data1 = pd.read_csv("./data/dev/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data2 = pd.read_csv("./data/test/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    labels1 = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels2 = pd.read_csv("./data/test/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels3 = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    assert len(data1) == len(labels1), "Dev set size mismatch"
    assert len(data2) == len(labels2), "Test set size mismatch"

    # get unique labels
    logging.info("Getting unique labels...")
    labels = pd.concat([labels1, labels2, labels3])
    unique_labels = set([label.strip().lower() for _, row in labels.iterrows() for label in row[0].split(" ")])

    # define model
    logging.info("Defining model...")
    # model = RobertaForSequenceClassification.from_pretrained(models_path)
    pipe = TextClassificationPipeline(
        model=RobertaForSequenceClassification.from_pretrained(models_path),
        tokenizer=tokenizer,
        return_all_scores=True,
    )

    print(tokenizer("Hello!"))

    for id, text in data1.iterrows():
        predictions = pipe("Hello!")
        print(predictions)
        break

    # # predict dev set
    # logging.info("Predicting dev set...")
    # dev_preds = model(dev_ds).predictions
    # test_preds = model(test_ds).predictions

