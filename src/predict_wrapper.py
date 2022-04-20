import argparse
import logging
import sys
from itertools import repeat
import fasttext
import pandas as pd

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description='Predict scores')
    parser.add_argument('stdin', nargs='?', type=argparse.FileType('r'), default=sys.stdin)

    args = parser.parse_args()
    stdin_data = args.stdin.read()

    # load saved model
    model = fasttext.load_model("./models/fasttext/fasttext.bin")

    # Make unique labels
    logging.info('Making unique labels...')
    unique_labels_tsv = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    unique_labels = unique_labels_tsv[0].tolist()

    label2idx = {}
    for label in unique_labels:
        label2idx[label] = len(label2idx)

    # delete empty lines
    stdin_data = stdin_data.strip()

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
