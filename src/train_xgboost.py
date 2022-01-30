import json
import logging
import os
import pickle

import pandas as pd
import yaml

from SKLearnDS import SKlearnDS

import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['random_forest']

    # Load the data
    logging.info('Loading data...')
    data1 = pd.read_csv("./data/dev/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data2 = pd.read_csv("./data/test/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data3 = pd.read_csv("./data/train/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    labels1 = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels2 = pd.read_csv("./data/test/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels3 = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    # Make unique labels
    logging.info('Making unique labels...')
    labels = pd.concat([labels1, labels2, labels3])
    unique_labels = set([label.strip().lower() for _, row in labels.iterrows() for label in row[0].split(" ")])

    label2idx = {}
    for label in unique_labels:
        label2idx[label] = len(label2idx)

    # Build datasets
    logging.info('Building datasets...')
    ds_train = SKlearnDS(data3[0], labels3, label2idx)
    ds_test = SKlearnDS(data2[0], labels2, label2idx)
    ds_dev = SKlearnDS(data1[0], labels1, label2idx)
    ds_texts, ds_labels = ds_train.get()

    # Train the model
    logging.info('Training the model...')
    boost = OneVsRestClassifier(xgb.XGBClassifier(objective='binary:logistic')).fit(ds_texts, ds_labels)

    # Predict
    logging.info('Predicting...')
    outs_test = boost.predict(ds_train.count_vect.transform(data2[0]))
    outs_dev = boost.predict(ds_train.count_vect.transform(data1[0]))

    # Evaluate
    logging.info('Evaluating...')
    acc_test = accuracy_score(ds_test.X_train_labels, outs_test)
    acc_dev = accuracy_score(ds_dev.X_train_labels, outs_dev)
    logging.info('Accuracy on test set: %.2f%%' % (acc_test * 100))
    logging.info('Accuracy on dev set: %.2f%%' % (acc_dev * 100))

    f1_test = f1_score(y_true=ds_test.X_train_labels, y_pred=outs_test, average='weighted')
    f1_dev = f1_score(y_true=ds_dev.X_train_labels, y_pred=outs_dev, average='weighted')
    logging.info('F1 score on test set: %.2f%%' % (f1_test * 100))
    logging.info('F1 score on dev set: %.2f%%' % (f1_dev * 100))

    precision_test = precision_score(y_true=ds_test.X_train_labels, y_pred=outs_test, average='weighted')
    precision_dev = precision_score(y_true=ds_dev.X_train_labels, y_pred=outs_dev, average='weighted')
    logging.info('Precision score on test set: %.2f%%' % (precision_test * 100))
    logging.info('Precision score on dev set: %.2f%%' % (precision_dev * 100))

    recall_test = recall_score(y_true=ds_test.X_train_labels, y_pred=outs_test, average='weighted')
    recall_dev = recall_score(y_true=ds_dev.X_train_labels, y_pred=outs_dev, average='weighted')
    logging.info('Recall score on test set: %.2f%%' % (recall_test * 100))
    logging.info('Recall score on dev set: %.2f%%' % (recall_dev * 100))

    with open("scores_classification.json", "w", encoding="utf8") as f:
        json.dump({
            "f1": f1_test,
            "accuracy": acc_test,
            "precision": precision_test,
            "recall": recall_test
        }, f, indent=4)

    # Save predictions to tsvs
    logging.info('Saving predictions...')
    translated_dev_preds = [" ".join(ds_dev.tensor2labels(model_outputs, unique_labels)) for model_outputs in (outs_dev >= 0.5).astype(int)]
    translated_test_preds = [" ".join(ds_dev.tensor2labels(model_outputs, unique_labels)) for model_outputs in (outs_test >= 0.5).astype(int)]

    # save predictions to csv file
    logging.info("Saving predictions to csv file...")
    with open("./data/dev/out.tsv", "w", encoding="utf8") as f:
        for pred in translated_dev_preds:
            f.write(pred + "\n")

    with open("./data/test/out.tsv", "w", encoding="utf8") as f:
        for pred in translated_test_preds:
            f.write(pred + "\n")

    # Save the model
    logging.info('Saving the model...')
    os.makedirs("./models/xgboost", exist_ok=True)
    with open("./models/xgboost/model.pkl", "wb") as f:
        pickle.dump(boost, f)
