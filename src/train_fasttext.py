import json
import logging
import os
import pickle

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

    # Load the data
    logging.info('Loading data...')
    data1 = pd.read_csv("./data/dev/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data2 = pd.read_csv("./data/test/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data3 = pd.read_csv("./data/train/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    labels1 = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels2 = pd.read_csv("./data/test/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels3 = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    # Save data for fasttext
    logging.info('Saving data for fasttext...')
    with open("./data/train/fasttext.txt", "w", encoding="utf8") as f:
        for inpt, labels in zip(data3[0], labels3[0]):
            new_labels = " ".join(["__label__" + label for label in labels.split(" ")])

            f.write(new_labels + " " + inpt + "\n")

    # Make unique labels
    logging.info('Making unique labels...')
    labels = pd.concat([labels1, labels2, labels3])
    unique_labels = set([label.strip().lower() for _, row in labels.iterrows() for label in row[0].split(" ")])

    label2idx = {}
    for label in unique_labels:
        label2idx[label] = len(label2idx)

    # Train the model
    logging.info('Training the model...')
    model = fasttext.train_supervised(input="./data/train/fasttext.txt", loss='ova', epoch=config['epochs'], lr=config['lr'])

    # Predict
    logging.info('Predicting...')

    dev_preds = []
    dev_gt = []
    translated_dev_preds = []
    dev_label_probs = []
    for item, lab in zip(data1[0], labels1[0]):
        preds = model.predict(item, k=len(label2idx))

        true_labels = []
        probes = []
        for label_name, label_weight in zip(preds[0], preds[1]):
            label_name = label_name.replace("__label__", "")
            if label_weight >= 0.5:
                true_labels.append(label_name)
            probes.append("{}:{:.9f}".format(label_name, label_weight))
        dev_label_probs.append(probes)

        pred_rewrited_labels = [label for label in true_labels]
        translated_dev_preds.append(" ".join(pred_rewrited_labels))

        pred_tensor = labels2tensor(pred_rewrited_labels, label2idx)
        gt_tensor = labels2tensor(lab.split(" "), label2idx)

        dev_preds.append(pred_tensor)
        dev_gt.append(gt_tensor)

    test_preds = []
    test_gt = []
    translated_test_preds = []
    test_label_probs = []
    for item, lab in zip(data2[0], labels2[0]):
        preds = model.predict(item, k=len(label2idx))
        # print(preds)
        # pred_labels = preds[0]

        true_labels = []
        probes = []
        for label_name, label_weight in zip(preds[0], preds[1]):
            label_name = label_name.replace("__label__", "")
            if label_weight >= 0.5:
                true_labels.append(label_name)
            probes.append("{}:{:.9f}".format(label_name, label_weight))
        test_label_probs.append(probes)

        pred_rewrited_labels = [label.replace("__label__", "") for label in true_labels]
        translated_test_preds.append(" ".join(pred_rewrited_labels))

        pred_tensor = labels2tensor(pred_rewrited_labels, label2idx)
        gt_tensor = labels2tensor(lab.split(" "), label2idx)

        test_preds.append(pred_tensor)
        test_gt.append(gt_tensor)

    # Evaluate
    logging.info('Evaluating...')
    acc_test = accuracy_score(test_gt, test_preds)
    acc_dev = accuracy_score(dev_gt, dev_preds)
    logging.info('Accuracy on test set: %.2f%%' % (acc_test * 100))
    logging.info('Accuracy on dev set: %.2f%%' % (acc_dev * 100))

    f1_test = f1_score(y_true=test_gt, y_pred=test_preds, average='weighted')
    f1_dev = f1_score(y_true=dev_gt, y_pred=dev_preds, average='weighted')
    logging.info('F1 score on test set: %.2f%%' % (f1_test * 100))
    logging.info('F1 score on dev set: %.2f%%' % (f1_dev * 100))

    precision_test = precision_score(y_true=test_gt, y_pred=test_preds, average='weighted')
    precision_dev = precision_score(y_true=dev_gt, y_pred=dev_preds, average='weighted')
    logging.info('Precision score on test set: %.2f%%' % (precision_test * 100))
    logging.info('Precision score on dev set: %.2f%%' % (precision_dev * 100))

    recall_test = recall_score(y_true=test_gt, y_pred=test_preds, average='weighted')
    recall_dev = recall_score(y_true=dev_gt, y_pred=dev_preds, average='weighted')
    logging.info('Recall score on test set: %.2f%%' % (recall_test * 100))
    logging.info('Recall score on dev set: %.2f%%' % (recall_dev * 100))

    with open("scores_classification.json", "w", encoding="utf8") as f:
        json.dump({
            "f1": f1_test,
            "accuracy": acc_test,
            "precision": precision_test,
            "recall": recall_test
        }, f, indent=4)

    if config['outputs'] == 'labels':
        # save predictions to csv file
        logging.info("Saving predictions to csv file...")
        with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
            for pred in translated_dev_preds:
                f.write(pred + "\n")

        with open("./data/test/out.tsv", "w", encoding="utf8") as f:
            for pred in translated_test_preds:
                f.write(pred + "\n")
    else:
        logging.info("Saving probabilities to csv file...")
        with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
            for pred in dev_label_probs:
                f.write(" ".join(pred) + "\n")

        with open("./data/test/out.tsv", "w", encoding="utf8") as f:
            for pred in test_label_probs:
                f.write(" ".join(pred) + "\n")

    # Save the model
    logging.info('Saving the model...')
    os.makedirs("./models/fasttext", exist_ok=True)
    model.save_model("./models/fasttext/fasttext.bin")
