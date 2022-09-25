import json
import logging
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, \
    TrainingArguments

from ClassificationDataset import ClassificationDataset

from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

# os.environ["WANDB_DISABLED"] = "true"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = (pred.predictions >= 0.5).astype(int)  # .argmax(-1)

    # print(labels, preds)

    # try:
    acc = accuracy_score(labels, preds)
    # except ValueError:

    return {
        'accuracy': acc,
        'f1': f1_score(y_true=labels, y_pred=preds, average='weighted'),
        'precision': precision_score(y_true=labels, y_pred=preds, average='weighted'),
        'recall': recall_score(y_true=labels, y_pred=preds, average='weighted')
    }


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['classification_train']
    # config_train = yaml.safe_load(open("./params.yaml"))['language_modeling_train']
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in config['cuda_visible_devices']])
    os.environ["WANDB_PROJECT"] = 'PetraRQ-Classifier-Books'

    # set random state
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # log to wandb
    logging.info("Logging to wandb...")
    wandb.login()

    # setup datasets paths
    dev_ds = "./data/dev/"
    train_ds = "./data/train/"

    # set models path
    # lm_model_path = "./models/roberta_lm"
    models_path = "./models/roberta_classifier_books"
    os.makedirs(models_path, exist_ok=True)

    logging.info("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("xlm-roberta-base", max_len=514,
                                                     use_fast=True)
    # Load the data
    logging.info('Loading data...')
    data_train = pd.read_csv("./data/train/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data_dev = pd.read_csv("./data/dev/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_train = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_dev = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

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

    # create datasets
    logging.info("Creating datasets...")
    dev_ds = ClassificationDataset(data_dev, labels_dev, unique_labels, tokenizer)
    train_ds = ClassificationDataset(data_train, labels_train, unique_labels, tokenizer)

    num_labels = len(unique_labels)

    # define model
    logging.info("Defining model...")
    model = RobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)

    # build trainer
    logging.info("Building trainer...")

    training_args = TrainingArguments(
        output_dir=models_path,
        warmup_steps=config['warmup_steps'],
        overwrite_output_dir=True,
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        save_steps=5_000,
        save_total_limit=3,
        do_train=True,
        do_eval=True,
        no_cuda=False,
        logging_steps=700,
        eval_steps=700,
        evaluation_strategy='steps',
        #report_to="wandb",
        run_name="petrarq-roberta-classification"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics
    )

    # train model
    logging.info("Training model...")
    trainer.train()

    # evaluate model
    logging.info("Evaluating model...")
    eval_scores = trainer.evaluate(dev_ds)

    scores = {
        'f1': eval_scores['eval_f1'],
        'accuracy': eval_scores['eval_accuracy'],
        'precision': eval_scores['eval_precision'],
        'recall': eval_scores['eval_recall'],
        'loss': eval_scores['eval_loss']
    }
    #
    # # predict dev set
    # logging.info("Predicting dev set...")
    # dev_preds = trainer.predict(dev_ds).predictions
    # test_preds = trainer.predict(test_ds).predictions
    #
    # translated_dev_preds = [" ".join(dev_ds.tensor2labels(model_outputs)) for model_outputs in (dev_preds >= 0.5).astype(int)]
    # translated_test_preds = [" ".join(test_ds.tensor2labels(model_outputs)) for model_outputs in (test_preds >= 0.5).astype(int)]
    #
    # # save predictions to csv file
    # logging.info("Saving predictions to csv file...")
    # with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
    #     for pred in translated_dev_preds:
    #         f.write(pred + "\n")
    #
    # with open("./data/test/out.tsv", "w", encoding="utf8") as f:
    #     for pred in translated_test_preds:
    #         f.write(pred + "\n")
    #
    # # save predictions
    # logging.info("Saving predictions...")
    #
    # # save model
    # logging.info("Saving model")
    trainer.save_model()

    # log results
    logging.info("Logging results...")
    with open("./scores_classification.json", "w", encoding="utf8") as f:
        json.dump(scores, f, indent=4)

    # wandb.finish()
