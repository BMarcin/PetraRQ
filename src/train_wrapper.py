import argparse
import logging
import os
import shutil
import lzma
import sys

import yaml
from pathlib import Path

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--seed',
        type=int,
        default=os.environ.get("SEED"),
        help='Random seed'
    )

    parser.add_argument(
        '--threads',
        type=int,
        default=os.environ.get("THREADS"),
        help='Number of threads to use'
    )

    args = parser.parse_args()

    seed = args.seed
    threads = args.threads

    # check if seed is set
    if not seed:
        parser.print_usage()
        sys.exit(1)

    # check if threads is set
    if not threads:
        parser.print_usage()
        sys.exit(1)

    logging.info(f'Using seed {seed} and {threads} threads')

    config_path = Path("./params.yaml")

    if not config_path.exists():
        logging.info(f'No config file found at {config_path}. Creating one...')
        config = {
            "logistic_regression": {
                "seed": seed,
                "threads": threads,
                "solver": "sag",
                "outputs": "probabilities"
            },
            "datasetrewrite": {
                "threads": threads
            }
        }

        logging.info(f'Saving config to {config_path.absolute()}')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    else:
        logging.info(f'Loading config from {config_path.absolute()}')

        config = yaml.safe_load(open(config_path))
        config["logistic_regression"]["seed"] = seed
        config["logistic_regression"]["threads"] = threads

        logging.info(f'Saving config to {config_path.absolute()}')
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    data_path = Path("./data/")
    if not data_path.exists():
        logging.info(f'No data folder found at {data_path}. Creating one...')
        data_path.mkdir(parents=True)

        # copy ./dev-0 to ./data/dev
        dev_path = Path("./dev-0/")
        if dev_path.exists():
            logging.info(f'Copying {dev_path.absolute()} to {data_path.absolute()}')
            shutil.copytree(dev_path, data_path / "dev")

            # for each file in ./data/dev if it ends with .xz decompress it
            # read and write in chunks of 1024 bytes
            for file in data_path.glob("dev/*.xz"):
                logging.info(f'Decompressing {file.absolute()}')
                with lzma.open(file, "rb") as f_in, open(file.with_suffix(""), "wb") as f_out:
                    for chunk in iter(lambda: f_in.read(1024), b""):
                        f_out.write(chunk)

        # copy ./test-A to ./data/test
        test_path = Path("./test-A/")
        if test_path.exists():
            logging.info(f'Copying {test_path.absolute()} to {data_path.absolute()}')
            shutil.copytree(test_path, data_path / "test")

            # for each file in ./data/test if it ends with .xz decompress it
            # read and write in chunks of 1024 bytes
            for file in data_path.glob("test/*.xz"):
                logging.info(f'Decompressing {file.absolute()}')
                with lzma.open(file, "rb") as f_in, open(file.with_suffix(""), "wb") as f_out:
                    for chunk in iter(lambda: f_in.read(1024), b""):
                        f_out.write(chunk)

        # copy ./train to ./data/train
        train_path = Path("./train/")
        if train_path.exists():
            logging.info(f'Copying {train_path.absolute()} to {data_path.absolute()}')
            shutil.copytree(train_path, data_path / "train")

            # for each file in ./data/train if it ends with .xz decompress it
            # read and write in chunks of 1024 bytes
            for file in data_path.glob("train/*.xz"):
                logging.info(f'Decompressing {file.absolute()}')
                with lzma.open(file, "rb") as f_in, open(file.with_suffix(""), "wb") as f_out:
                    for chunk in iter(lambda: f_in.read(1024), b""):
                        f_out.write(chunk)
