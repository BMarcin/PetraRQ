import json
import logging
import os

import pandas as pd
import wandb
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, \
    TrainingArguments

from src.ClassificationDataset import ClassificationDataset


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
    config_train = yaml.safe_load(open("./params.yaml"))['language_modeling_train']
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in config['cuda_visible_devices']])

    # log to wandb
    logging.info("Logging to wandb...")
    wandb.login()

    # setup datasets paths
    dev_ds = "./data/dev/"
    test_ds = "./data/test/"
    train_ds = "./data/train/"

    # set models path
    lm_model_path = "./models/roberta_lm"
    models_path = "./models/roberta_classifier"
    os.makedirs(models_path, exist_ok=True)

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

    # add special tokens
    tokenizer.add_special_tokens({
        'additional_special_tokens': special_tokens
    })

    # load datasets
    logging.info("Loading datasets...")
    data1 = pd.read_csv("./data/dev/in.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data2 = pd.read_csv("./data/test/in.tsv", delimiter='\t', header=None, encoding="utf8")
    data3 = pd.read_csv("./data/train/in.tsv", delimiter='\t', header=None, encoding="utf8")

    labels1 = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels2 = pd.read_csv("./data/test/expected.tsv", delimiter='\t', header=None, encoding="utf8")
    labels3 = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8")

    assert len(data1) == len(labels1), "Dev set size mismatch"
    assert len(data2) == len(labels2), "Test set size mismatch"
    assert len(data3) == len(labels3), "Train set size mismatch"

    # get unique labels
    logging.info("Getting unique labels...")
    labels = pd.concat([labels1, labels2, labels3])
    unique_labels = set([label for _, row in labels.iterrows() for label in row[0].split(" ")])

    # create datasets
    logging.info("Creating datasets...")
    dev_ds = ClassificationDataset(data1, labels1, unique_labels, tokenizer)
    test_ds = ClassificationDataset(data2, labels2, unique_labels, tokenizer)
    train_ds = ClassificationDataset(data3, labels3, unique_labels, tokenizer)

    num_labels = len(unique_labels)

    # define model
    logging.info("Defining model...")
    model = RobertaForSequenceClassification.from_pretrained(lm_model_path, num_labels=num_labels)

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
        report_to="wandb",
        run_name="roberta-classification"
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
    eval_scores = trainer.evaluate(test_ds)

    scores = {
        'f1': eval_scores['eval_f1'],
        'accuracy': eval_scores['eval_accuracy'],
        'precision': eval_scores['eval_precision'],
        'recall': eval_scores['eval_recall'],
        'loss': eval_scores['eval_loss']
    }

    # log results
    logging.info("Logging results...")
    with open("./scores_classification.json", "w", encoding="utf8") as f:
        json.dump(scores, f, indent=4)
