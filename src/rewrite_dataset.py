import logging

import pandas as pd
import yaml
from tqdm.auto import tqdm

from DatasetRewriter import process_datasets

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
    dev, test, train = process_datasets(
        dev_input_texts=dev_in[0].tolist(),
        test_input_texts=test_in[0].tolist(),
        train_input_texts=train_in[0].tolist(),
        threads=config['threads']
    )

    # with open("./data/dev/lm.txt", "w", encoding="utf8") as f:
    #     for item in tqdm(dev, desc="Saving dev ds"):
    #         f.write(item + '\n')
    #
    # with open("./data/test/lm.txt", "w", encoding="utf8") as f:
    #     for item in tqdm(test, desc="Saving test ds"):
    #         f.write(item + '\n')
    #
    # with open("./data/train/lm.txt", "w", encoding="utf8") as f:
    #     for item in tqdm(train, desc="Saving train ds"):
    #         f.write(item + '\n')

    dev_in[2] = dev
    test_in[2] = test
    train_in[2] = train

    dev_in[2].to_csv("./data/dev/processed.tsv", sep='\t', header=False, index=False, encoding="utf8")
    test_in[2].to_csv("./data/test/processed.tsv", sep='\t', header=False, index=False, encoding="utf8")
    train_in[2].to_csv("./data/train/processed.tsv", sep='\t', header=False, index=False, encoding="utf8")
