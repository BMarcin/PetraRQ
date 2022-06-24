import json
import logging
import pickle
from itertools import repeat
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from SKLearnDS import SKlearnDS

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['naive_bayes']

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

    ds_dev = SKlearnDS(data_dev[0], labels_dev, label2idx)
    if use_test_data:
        ds_test = SKlearnDS(data_test[0], labels_test, label2idx)

    # Load models
    logging.info('Loading models...')
    with open("./models/naive_bayes/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("./models/naive_bayes/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("./models/naive_bayes/labels.pkl", "rb") as f:
        labels = pickle.load(f)

    # Predict
    logging.info('Predicting...')
    outs_dev = model.predict(vectorizer.transform(data_dev[0]))
    outs_test = model.predict(vectorizer.transform(data_test[0]))

    # Evaluate
    logging.info('Evaluating...')
    acc_dev = accuracy_score(ds_dev.X_train_labels, outs_dev)
    logging.info('Accuracy on dev set: {}'.format(acc_dev))
    if use_test_data:
        acc_test = accuracy_score(ds_test.X_train_labels, outs_test)
        logging.info('Accuracy on test set: {}'.format(acc_test))

    f1_dev = f1_score(y_true=ds_dev.X_train_labels, y_pred=outs_dev, average='weighted')
    logging.info('F1 score on dev set: {}'.format(f1_dev))
    if use_test_data:
        f1_test = f1_score(y_true=ds_test.X_train_labels, y_pred=outs_test, average='weighted')
        logging.info('F1 score on test set: {}'.format(f1_test))

    precision_dev = precision_score(y_true=ds_dev.X_train_labels, y_pred=outs_dev, average='weighted')
    logging.info('Precision score on dev set: {}'.format(precision_dev))
    if use_test_data:
        precision_test = precision_score(y_true=ds_test.X_train_labels, y_pred=outs_test, average='weighted')
        logging.info('Precision score on test set: {}'.format(precision_test))

    recall_dev = recall_score(y_true=ds_dev.X_train_labels, y_pred=outs_dev, average='weighted')
    logging.info('Recall score on dev set: {}'.format(recall_dev))
    if use_test_data:
        recall_test = recall_score(y_true=ds_test.X_train_labels, y_pred=outs_test, average='weighted')
        logging.info('Recall score on test set: {}'.format(recall_test))

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
        translated_dev_preds = [" ".join(ds_dev.tensor2labels(model_outputs, unique_labels)) for model_outputs in (outs_dev >= 0.5).astype(int)]
        translated_test_preds = [" ".join(ds_test.tensor2labels(model_outputs, unique_labels)) for model_outputs in (outs_test >= 0.5).astype(int)]

        # save predictions to csv file
        logging.info('Saving predictions...')
        with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
            for pred in translated_dev_preds:
                f.write(pred + "\n")

        with open("./data/test/out.tsv", "w", encoding="utf8") as f:
            for pred in translated_test_preds:
                f.write(pred + "\n")
    else:
        # Predict proba
        logging.info('Predicting probabilities...')
        outs_dev_proba = model.predict_proba(vectorizer.transform(data_dev[0])).astype(float)
        outs_test_proba = model.predict_proba(vectorizer.transform(data_test[0])).astype(float)

        # Translate predictions
        logging.info('Translate predictions...')
        translated_dev_preds = []
        for probabilities, labels in zip(outs_dev_proba, repeat(unique_labels)):
            score_lines = []
            for prob, label in zip(probabilities, labels):
                # adjust for loglikelihood metric
                prob = prob if prob <= 1 - float(config['epsilon']) else 1 - float(config['epsilon'])
                prob = prob if prob >= 0 + float(config['epsilon']) else 0 + float(config['epsilon'])
                score_lines.append("{}:{:.9f}".format(label, prob))
            translated_dev_preds.append(" ".join(score_lines))

        translated_test_preds = []
        for probabilities, labels in zip(outs_test_proba, repeat(unique_labels)):
            score_lines = []
            for prob, label in zip(probabilities, labels):
                # adjust for loglikelihood metric
                prob = prob if prob <= 1 - float(config['epsilon']) else 1 - float(config['epsilon'])
                prob = prob if prob >= 0 + float(config['epsilon']) else 0 + float(config['epsilon'])
                score_lines.append("{}:{:.9f}".format(label, prob))
            translated_test_preds.append(" ".join(score_lines))

        # save predictions to csv file
        logging.info('Saving predictions...')
        with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
            for pred in translated_dev_preds:
                f.write(pred + "\n")

        with open("./data/test/out.tsv", "w", encoding="utf8") as f:
            for pred in translated_test_preds:
                f.write(pred + "\n")

    logging.info('Done!')
