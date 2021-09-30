import csv
import json
import logging
from pathlib import Path

from tqdm.auto import tqdm

from DatasetSplitter import dataset_splitter_by_time


def save_ds_part(items: list, in_filename: Path, expected_filename: Path):
    with open(in_filename, "w", encoding="utf8") as in_file:
        with open(expected_filename, "w", encoding="utf8") as expected_file:
            csv_in_writer = csv.writer(in_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            csv_expected_writer = csv.writer(expected_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)

            for item in tqdm(items, desc='Saving {} and {}'.format(str(in_filename), str(expected_filename))):
                csv_in_writer.writerow([item['date'], item['content']])
                csv_expected_writer.writerow([','.join(item['buckets'])])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    dev_ds_size = 0.1
    test_ds_size = 0.2
    train_ds_size = 0.7

    parts = 3

    ds_path = Path("./data/parsed-pdfs.json")
    if ds_path.exists():
        logging.info("Reading input file")
        data = json.loads(ds_path.read_text(encoding="utf8"))

        logging.info('Splitting dataset')
        splitted_items = dataset_splitter_by_time(
            data,
            dev_ds_size,
            test_ds_size,
            train_ds_size,
            parts
        )

        logging.info('Saving dataset to splitted files')
        save_ds_part(splitted_items['dev'], Path('./data/dev/in.tsv'), Path('./data/dev/expected.tsv'))
        save_ds_part(splitted_items['test'], Path('./data/test/in.tsv'), Path('./data/test/expected.tsv'))
        save_ds_part(splitted_items['train'], Path('./data/train/in.tsv'), Path('./data/train/expected.tsv'))
    else:
        logging.error("File {} does not exists".format(str(ds_path)))
