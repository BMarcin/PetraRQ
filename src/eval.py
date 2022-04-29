import json
import logging
import os
import pickle
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from transformers import RobertaTokenizerFast, TextClassificationPipeline, RobertaForSequenceClassification


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
    config = yaml.safe_load(open("./params.yaml"))['classification_train']
    config_train = yaml.safe_load(open("./params.yaml"))['language_modeling_train']
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in config['cuda_visible_devices']])

    # Check if we can use Test data
    use_test_data = False
    if Path("./data/test/expected.tsv").exists():
        logging.info("Found test data, using it for evaluation.")
        use_test_data = True

    # Load the data
    logging.info('Loading data...')
    data_dev = pd.read_csv("./data/dev/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_dev = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    data_test = pd.read_csv("./data/test/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    if use_test_data:
        labels_test = pd.read_csv("./data/test/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    # Make unique labels
    logging.info('Making unique labels...')
    unique_labels_tsv = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    unique_labels = unique_labels_tsv[0].tolist()

    label2idx = {}
    for label in unique_labels:
        label2idx[label] = len(label2idx)

    # Load models
    logging.info('Loading models...')
    lm_model_path = "./models/roberta_lm"
    models_path = "./models/roberta_classifier"

    logging.info("Defining special characters...")
    special_tokens = [
        '<url>',
        '<email>',
        '<number>',
        '<date>',
    ]

    tokenizer = RobertaTokenizerFast.from_pretrained(
        lm_model_path,
        use_fast=True)

    logging.info("Defining model...")
    pipe = TextClassificationPipeline(
        model=RobertaForSequenceClassification.from_pretrained(models_path, ),
        tokenizer=tokenizer,
        return_all_scores=True,
        max_length=config_train['max_seq_length'],
        truncation=True,
    )

    # Predict
    logging.info('Predicting...')
    dev_probabilities = []
    dev_predicted_labels = []
    dev_gt = []
    dev_preds = []

    dev_predictions = pipe(list(data_dev[0]), batch_size=config['per_device_eval_batch_size'])
    # for probe, gt_labels in tqdm(zip(dev_predictions, labels_dev[0]), desc="Predicting dev set"):
    for i in tqdm(range(len(dev_predictions)), desc="Predicting dev set"):
        probe = dev_predictions[i]
        gt_labels = labels_dev[0][i]

        labels_probabilities = []
        text_labels = []
        preds = []
        for label, label_name in zip(probe, unique_labels):
            score = label['score'] if label['score'] <= 1.0 else 1.0
            labels_probabilities.append("{}:{:.9f}".format(label_name, score))
            preds.append("{:.9f}".format(score))
            if label['score'] >= 0.5:
                text_labels.append(label_name)
        dev_preds.append(preds)
        dev_probabilities.append(" ".join(labels_probabilities))
        dev_predicted_labels.append(" ".join(text_labels))
        gt_tensor = labels2tensor(gt_labels.split(" "), label2idx)
        dev_gt.append(gt_tensor)


    test_probabilities = []
    test_predicted_labels = []
    test_gt = []
    test_preds = []

    test_predictions = pipe(list(data_test[0]), batch_size=config['per_device_eval_batch_size'])

    for i in tqdm(range(len(test_predictions)), desc="Predicting test set"):
        probe = test_predictions[i]

        if use_test_data:
            gt_labels = labels_test[0][i]

        labels_probabilities = []
        text_labels = []
        preds = []
        for label, label_name in zip(probe, unique_labels):
            score = label['score'] if label['score'] <= 1.0 else 1.0
            labels_probabilities.append("{}:{:.9f}".format(label_name, score))
            preds.append("{:.9f}".format(score))
            if label['score'] >= 0.5:
                text_labels.append(label_name)
        test_probabilities.append(" ".join(labels_probabilities))
        test_predicted_labels.append(" ".join(text_labels))
        if use_test_data:
            gt_tensor = labels2tensor(gt_labels.split(" "), label2idx)

    # Evaluate
    logging.info('Evaluating...')
    acc_dev = accuracy_score(dev_gt, dev_preds)
    logging.info('Accuracy on dev set: %.2f%%' % (acc_dev * 100))
    if use_test_data:
        acc_test = accuracy_score(test_gt, test_preds)
        logging.info('Accuracy on test set: %.2f%%' % (acc_test * 100))

    f1_dev = f1_score(y_true=dev_gt, y_pred=dev_preds, average='macro')
    logging.info('F1 on dev set: %.2f%%' % (f1_dev * 100))
    if use_test_data:
        f1_test = f1_score(y_true=test_gt, y_pred=test_preds, average='macro')
        logging.info('F1 on test set: %.2f%%' % (f1_test * 100))

    precision_dev = precision_score(y_true=dev_gt, y_pred=dev_preds, average='macro')
    logging.info('Precision on dev set: %.2f%%' % (precision_dev * 100))
    if use_test_data:
        precision_test = precision_score(y_true=test_gt, y_pred=test_preds, average='macro')
        logging.info('Precision on test set: %.2f%%' % (precision_test * 100))

    recall_dev = recall_score(y_true=dev_gt, y_pred=dev_preds, average='macro')
    logging.info('Recall on dev set: %.2f%%' % (recall_dev * 100))
    if use_test_data:
        recall_test = recall_score(y_true=test_gt, y_pred=test_preds, average='macro')
        logging.info('Recall on test set: %.2f%%' % (recall_test * 100))

    logging.info('Saving results...')
    if use_test_data:
        with open("scores_classification.json", "w", encoding="utf8") as f:
            json.dump({
                "f1": f1_test,
                "accuracy": acc_test,
                "precision": precision_test,
                "recall": recall_test
            }, f, indent=4)
    else:
        with open("scores_classification.json", "w", encoding="utf8") as f:
            json.dump({
                "f1": f1_dev,
                "accuracy": acc_dev,
                "precision": precision_dev,
                "recall": recall_dev
            }, f, indent=4)

        # Save ds predictions outputs
        if config['outputs'] == 'labels':
            # save predictions to csv file
            logging.info("Saving predictions to csv file...")
            with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
                for pred in dev_probabilities:
                    f.write(pred + "\n")

            # if use_test_data:
            with open("./data/test/out.tsv", "w", encoding="utf8") as f:
                for pred in test_probabilities:
                    f.write(pred + "\n")
        else:
            logging.info("Saving probabilities to csv file...")
            with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
                for pred in dev_preds:
                    f.write(" ".join(pred) + "\n")

            # if use_test_data:
            logging.info("Saving test probabilities to csv file...")
            with open("./data/test/out.tsv", "w", encoding="utf8") as f:
                for pred in test_preds:
                    f.write(" ".join(pred) + "\n")

        logging.info('Done!')
