import argparse
import logging
import sys
from itertools import repeat
import pandas as pd
import yaml
from tqdm.auto import tqdm
from transformers import RobertaTokenizerFast, TextClassificationPipeline, RobertaForSequenceClassification

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description='Predict scores')
    parser.add_argument('stdin', nargs='?', type=argparse.FileType('r'), default=sys.stdin)

    args = parser.parse_args()
    stdin_data = args.stdin.read()

    # Load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['classification_train']
    config_train = yaml.safe_load(open("./params.yaml"))['language_modeling_train']

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

    # Make unique labels
    logging.info('Making unique labels...')
    unique_labels_tsv = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    unique_labels = unique_labels_tsv[0].tolist()

    label2idx = {}
    for label in unique_labels:
        label2idx[label] = len(label2idx)

    # delete empty lines
    stdin_data = stdin_data.strip()

    predictions = pipe(stdin_data.split('\n'), batch_size=config['per_device_eval_batch_size'])
    for i in tqdm(range(len(predictions)), desc="Predicting dev set"):
        probe = predictions[i]

        for label, label_name in zip(probe, unique_labels):
            labels_probabilities.append("{}:{:.9f}".format(label_name, label['score']))
            if label['score'] >= 0.5:
                text_labels.append(label_name)


    pred_probes = []
    for input_line in stdin_data.split('\n'):
        prediction = model.predict(input_line, k=len(label2idx))

        true_labels = []
        probes = []
        for label_name, label_weight in zip(prediction[0], prediction[1]):
            label_name = label_name.replace("__label__", "")
            if label_weight >= 0.5:
                true_labels.append(label_name)
            if label_weight > 1:
                label_weight = 1
            probes.append("{}:{:.9f}".format(label_name, label_weight))
        pred_probes.append(" ".join(probes))

    # print results
    print("\n".join(pred_probes))
