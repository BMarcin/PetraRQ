import logging

import pandas as pd
import yaml

from DatasetRewriter import rewrite_datasets_to_txt

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    config = yaml.safe_load(open("./params.yaml"))['datasetrewrite']

    logging.info("Reading dev dataset")
    dev_in = pd.read_csv("./data/dev/in.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")

    logging.info("Reading test dataset")
    test_in = pd.read_csv("./data/test/in.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")

    logging.info("Reading train dataset")
    train_in = pd.read_csv("./data/train/in.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")

    logging.info("Rewriting")
    rewrite_datasets_to_txt(
        dev_input_texts=dev_in[0].tolist(),
        test_input_texts=test_in[0].tolist(),
        train_input_texts=train_in[0].tolist(),
        threads=config['threads']
    )
