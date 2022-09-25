import json
import logging
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
from transformers import RobertaTokenizerFast, RobertaModel

from PetraRQ import PetraRQ, coll_fn, ClassificationDataset


def labels2tensor(labels, label2idx):
    # print(labels)
    unique = set([label2idx[label.strip().lower()] for label in labels])
    if len(unique) == 0:
        return torch.zeros([len(label2idx)], dtype=torch.long).tolist()
    return torch.zeros([len(label2idx)]).index_fill_(0, torch.tensor(list(unique)), 1).tolist()


def process_texts(texts):
    # tokenized_texts = tokenizer.batch_encode_plus(
    #     texts,
    # )

    processed_texts = []
    for text in texts:
        tokenized_text = tokenizer.encode(text)
        collfunc = coll_fn([
            (tokenized_text,
             [0],
             524288)
        ])
        processed_texts.append(
            (torch.tensor(collfunc[0]), torch.tensor(collfunc[2]))
        )
    return processed_texts


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['classification_train']
    config_train = yaml.safe_load(open("./params.yaml"))['language_modeling_train']
    config_eval = yaml.safe_load(open("./params.yaml"))['classification_eval']
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
    models_path = "./models/petrarq_classifier_books"

    tokenizer = RobertaTokenizerFast.from_pretrained(
        "xlm-roberta-base",
        use_fast=True)

    logging.info("Defining model...")
    model = RobertaModel.from_pretrained("xlm-roberta-base")
    embeds = model.embeddings

    petra = PetraRQ(
        d_model=config_train['hidden_size'],
        num_labels=len(unique_labels),
        seq_length=config['seq_length'],
        overlapping_part=config['overlapping_part'],
        embeddings=embeds,
        model=model,
        lr=float(config['lr']),
    ).to("cuda")
    point = torch.load(
        os.path.join(models_path, "pytorch_model.ckpt"),
    )
    petra.load_state_dict(point["state_dict"])
    petra.eval()

    # Predict
    logging.info('Predicting...')
    dev_probabilities = []
    dev_predicted_labels = []
    dev_gt = []
    dev_preds = []
    dev_raw_preds = []

    dev_ins = process_texts(list(data_dev[0]))

    # dev_predictions = pipe(list(data_dev[0]), batch_size=config['per_device_eval_batch_size'])
    # for probe, gt_labels in tqdm(zip(dev_predictions, labels_dev[0]), desc="Predicting dev set"):
    with torch.no_grad():
        for i, (tensor, attention) in enumerate(tqdm(dev_ins, desc="Predicting dev set")):
            # print(tensor.shape)
            probe = petra(tensor, attention)
            # print(probe)
            gt_labels = labels_dev[0][i]
            # print(gt_labels)

            labels_probabilities = []
            text_labels = []
            preds = []
            raw_preds = []
            for label, label_name in zip(probe.detach().cpu().tolist()[0], unique_labels):
                score = label if label <= 1.0 else 1.0

                score = score if score <= 1 - float(config_eval['epsilon']) else 1 - float(config_eval['epsilon'])
                score = score if score >= 0 + float(config_eval['epsilon']) else 0 + float(config_eval['epsilon'])

                labels_probabilities.append(label_name)
                raw_preds.append("{}:{:.9f}".format(label_name, score))
                if label >= 0.5:
                    text_labels.append(label_name)
                    preds.append(1)
                else:
                    preds.append(0)
            dev_preds.append(preds)
            dev_raw_preds.append(raw_preds)
            dev_probabilities.append(" ".join(labels_probabilities))
            dev_predicted_labels.append(" ".join(text_labels))
            gt_tensor = labels2tensor(gt_labels.split(" "), label2idx)
            dev_gt.append(gt_tensor)

    test_probabilities = []
    test_predicted_labels = []
    test_gt = []
    test_preds = []
    test_raw_preds = []

    dev_ins = process_texts(list(data_test[0]))

    with torch.no_grad():
        for i, (tensor, attention) in enumerate(tqdm(dev_ins, desc="Predicting test set")):
            probe = petra(tensor, attention)

            if use_test_data:
                gt_labels = labels_test[0][i]

            labels_probabilities = []
            text_labels = []
            preds = []
            raw_preds = []
            for label, label_name in zip(probe.detach().cpu().tolist()[0], unique_labels):
                score = label if label <= 1.0 else 1.0

                score = score if score <= 1 - float(config_eval['epsilon']) else 1 - float(config_eval['epsilon'])
                score = score if score >= 0 + float(config_eval['epsilon']) else 0 + float(config_eval['epsilon'])

                labels_probabilities.append(label_name)
                # labels_probabilities.append("{}:{:.9f}".format(label_name, score))
                raw_preds.append("{}:{:.9f}".format(label_name, score))
                # raw_preds.append("{:.9f}".format(score))
                if label >= 0.5:
                    text_labels.append(label_name)
                    preds.append(1)
                else:
                    preds.append(0)
            test_probabilities.append(" ".join(labels_probabilities))
            test_predicted_labels.append(" ".join(text_labels))
            test_preds.append(preds)
            test_raw_preds.append(raw_preds)
            if use_test_data:
                gt_tensor = labels2tensor(gt_labels.split(" "), label2idx)
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
            for pred in dev_raw_preds:
                f.write(" ".join(pred) + "\n")

        # if use_test_data:
        logging.info("Saving test probabilities to csv file...")
        with open("./data/test/out.tsv", "w", encoding="utf8") as f:
            for pred in test_raw_preds:
                f.write(" ".join(pred) + "\n")

    logging.info('Done!')
