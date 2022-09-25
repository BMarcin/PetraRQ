import csv
import json
import logging
from dateutil.parser import parse
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime
import hashlib

from tqdm.auto import tqdm

from DatasetSplitter import dataset_splitter_by_time
import cleantext


def save_ds_part(items: list, in_filename: Path, expected_filename: Path):
    with open(in_filename, "w", encoding="utf8") as in_file:
        with open(expected_filename, "w", encoding="utf8") as expected_file:
            csv_in_writer = csv.writer(in_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
            csv_expected_writer = csv.writer(expected_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")

            for item in tqdm(items, desc='Saving {} and {}'.format(str(in_filename), str(expected_filename))):
                processed_item = cleantext.clean(
                    item['content'].replace("\n", "\\n").replace("\t", "\\t").replace("'", " ").replace('"', ' '),
                    fix_unicode=True,
                    lower=False,
                    normalize_whitespace=True,
                    no_line_breaks=True,
                    no_urls=True,
                    no_emails=True,
                    no_phone_numbers=True,
                    no_numbers=True,
                    no_digits=True,
                    no_currency_symbols=True,
                    no_punct=True,
                    no_emoji=True,
                    lang='en'
                )
                if len(processed_item.replace(" ", "")) >= 10:
                    csv_in_writer.writerow([processed_item, item['date']])
                    buckets = item['labels']
                    csv_expected_writer.writerow([' '.join(buckets)])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    config = yaml.safe_load(open("./params.yaml"))['datasetsplit']

    dev_ds_size = config['dev']
    test_ds_size = config['test']
    train_ds_size = config['train']

    parts = config['parts']

    ds_path = Path("./data/parsed_books.json")
    if ds_path.exists():
        logging.info("Reading input file")
        data_src = json.loads(ds_path.read_text(encoding="utf8"))

        logging.info("Reformatting data")
        data = []
        for item in tqdm(data_src):
            item['date'] = item['date'][:10]
            item['hash'] = hashlib.md5(item['content'].encode('utf-8')).hexdigest()
            data.append(item)

        # sort data by date
        data = sorted(data, key=lambda x: x[config['sort_by']])

        logging.info('Splitting dataset')
        splitted_items = dataset_splitter_by_time(
            data,
            dev_ds_size,
            test_ds_size,
            train_ds_size,
            parts
        )

        # labels = json.loads(Path("./data/labels_replacement_list.json").read_text(encoding="utf8"))

        logging.info('Saving dataset to splitted files')
        save_ds_part(splitted_items['dev'], Path('./data/dev/in.tsv'), Path('./data/dev/expected.tsv'))
        save_ds_part(splitted_items['test'], Path('./data/test/in.tsv'), Path('./data/test/expected.tsv'))
        save_ds_part(splitted_items['train'], Path('./data/train/in.tsv'), Path('./data/train/expected.tsv'))

        data_dev = pd.read_csv("./data/dev/in.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")
        labels_dev = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")

        data_test = pd.read_csv("./data/test/in.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")
        labels_test = pd.read_csv("./data/test/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")

        data_train = pd.read_csv("./data/train/in.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")
        labels_train = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0, quotechar="'")

        assert len(data_dev) == len(labels_dev), 'DEV dataset lines count missmatch, got {} and {}'.format(len(data_dev), len(labels_dev))
        assert len(data_test) == len(labels_test), 'TEST dataset lines count missmatch, got {} and {}'.format(len(data_test), len(labels_test))
        assert len(data_train) == len(labels_train), 'TRAIN dataset lines count missmatch, got {} and {}'.format(len(data_train), len(labels_train))
    else:
        logging.error("File {} does not exists".format(str(ds_path)))
