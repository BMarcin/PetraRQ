import argparse
import logging
import os
import sys
from itertools import repeat
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
from transformers import RobertaTokenizerFast, TextClassificationPipeline, RobertaForSequenceClassification, \
    RobertaModel

from PetraRQ import coll_fn, PetraRQ


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
             65536)
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
    models_path = "./models/petrarq_classifier"

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

    logging.info('Making unique labels...')
    unique_labels_tsv = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    unique_labels = unique_labels_tsv[0].tolist()

    logging.info("Defining model...")
    model = RobertaModel.from_pretrained(lm_model_path)
    embeds = model.embeddings

    petra = PetraRQ(
        d_model=config_train['hidden_size'],
        num_labels=len(unique_labels),
        seq_length=config['seq_length'],
        overlapping_part=config['overlapping_part'],
        steps=config['steps'],
        embeddings=embeds,
        model=model,
        lr=float(config['lr']),
    ).to("cuda")
    point = torch.load(
        os.path.join(models_path, "pytorch_model.ckpt"),
    )
    petra.load_state_dict(point["state_dict"])
    petra.eval()

    label2idx = {}
    for label in unique_labels:
        label2idx[label] = len(label2idx)

    # delete empty lines
    stdin_data = stdin_data.strip()

    # predictions = pipe(stdin_data.split('\n'), batch_size=config['per_device_eval_batch_size'])
    dev_ins = process_texts([stdin_data])

    with torch.no_grad():
        probabilities = []
        for i, (tensor, attention) in enumerate(tqdm(dev_ins, desc="Predicting....")):
            probe = petra(tensor, attention)

            labels_probabilities = []
            for label, label_name in zip(probe.detach().cpu().tolist()[0], unique_labels):
                score = label if label <= 1.0 else 1.0
                labels_probabilities.append("{}:{:.9f}".format(label_name, score))
            probabilities.append(" ".join(labels_probabilities))

        print("\n".join(probabilities))
