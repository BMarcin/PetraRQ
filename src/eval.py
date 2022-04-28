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
    dev_predictions = pipe(list(data_dev[0]), batch_size=config['per_device_eval_batch_size'])
    # for probe, gt_labels in tqdm(zip(dev_predictions, labels_dev[0]), desc="Predicting dev set"):
    for i in tqdm(range(len(dev_predictions)), desc="Predicting dev set"):
        probe = dev_predictions[i]
        gt_labels = labels_dev[0][i]

        labels_probabilities = []
        text_labels = []
        for label, label_name in zip(probe, unique_labels):
            labels_probabilities.append("{}:{:.9f}".format(label_name, label['score']))
            if label['score'] >= 0.5:
                text_labels.append(label_name)
        dev_probabilities.append(" ".join(labels_probabilities))
        dev_predicted_labels.append(" ".join(text_labels))



    dev_preds = []
    dev_gt = []
    translated_dev_preds = []
    dev_label_probs = []

    for item, lab in zip(data_dev[0], labels_dev[0]):
        preds = model.predict(item, k=len(label2idx))

        true_labels = []
        probes = []
        for label_name, label_weight in zip(preds[0], preds[1]):
            label_name = label_name.replace("__label__", "")
            if label_weight >= 0.5:
                true_labels.append(label_name)
            if label_weight > 1:
                label_weight = 1
            probes.append("{}:{:.9f}".format(label_name, label_weight))
        dev_label_probs.append(probes)

        pred_rewrited_labels = [label for label in true_labels]
        translated_dev_preds.append(" ".join(pred_rewrited_labels))

        pred_tensor = labels2tensor(pred_rewrited_labels, label2idx)
        gt_tensor = labels2tensor(lab.split(" "), label2idx)

        dev_preds.append(pred_tensor)
        dev_gt.append(gt_tensor)

    # if use_test_data:
    test_preds = []
    test_gt = []
    translated_test_preds = []
    test_label_probs = []

    for item, lab in zip(data_test[0], labels_test[0]):
        preds = model.predict(item, k=len(label2idx))
        # print(preds)
        # pred_labels = preds[0]

        true_labels = []
        probes = []
        for label_name, label_weight in zip(preds[0], preds[1]):
            label_name = label_name.replace("__label__", "")
            if label_weight >= 0.5:
                true_labels.append(label_name)
            if label_weight > 1:
                label_weight = 1
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

    # save results to "scores_classification.json"
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
            for pred in translated_dev_preds:
                f.write(pred + "\n")

        # if use_test_data:
        with open("./data/test/out.tsv", "w", encoding="utf8") as f:
            for pred in translated_test_preds:
                f.write(pred + "\n")
    else:
        logging.info("Saving probabilities to csv file...")
        with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
            for pred in dev_label_probs:
                f.write(" ".join(pred) + "\n")

        # if use_test_data:
        logging.info("Saving test probabilities to csv file...")
        with open("./data/test/out.tsv", "w", encoding="utf8") as f:
            for pred in test_label_probs:
                f.write(" ".join(pred) + "\n")

    logging.info('Done!')
