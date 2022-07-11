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
        help='Number of threads'
    )

    parser.add_argument(
        '--gpus',
        type=int,
        default=os.environ.get("GPUS"),
        help='Number of threads to use'
    )

    parser.add_argument(
        '--lm_batch_size',
        type=int,
        default=os.environ.get("LM_BATCH_SIZE"),
        help='Batch size for language model'
    )

    parser.add_argument(
        '--lm_epochs',
        type=int,
        default=os.environ.get("LM_EPOCHS"),
        help='Number of epochs for language model'
    )

    parser.add_argument(
        '--classifier_batch_size',
        type=int,
        default=os.environ.get("CLASSIFIER_BATCH_SIZE"),
        help='Batch size for classifier'
    )

    parser.add_argument(
        '--classifier_epochs',
        type=int,
        default=os.environ.get("CLASSIFIER_EPOCHS"),
        help='Number of epochs for classifier'
    )

    args = parser.parse_args()

    seed = args.seed
    gpus = args.gpus
    threads = args.threads
    lm_batch_size = args.lm_batch_size
    lm_epochs = args.lm_epochs
    classifier_batch_size = args.classifier_batch_size
    classifier_epochs = args.classifier_epochs

    # check if seed is set
    if not seed:
        parser.print_usage()
        sys.exit(1)

    # check if gpus is set
    if not gpus:
        parser.print_usage()
        sys.exit(1)

    # check if threads is set
    if not threads:
        parser.print_usage()
        sys.exit(1)

    # check if lm_batch_size is set
    if not lm_batch_size:
        parser.print_usage()
        sys.exit(1)

    # check if lm_epochs is set
    if not lm_epochs:
        parser.print_usage()
        sys.exit(1)

    # check if classifier_batch_size is set
    if not classifier_batch_size:
        parser.print_usage()
        sys.exit(1)

    # check if classifier_epochs is set
    if not classifier_epochs:
        parser.print_usage()
        sys.exit(1)

    logging.info(f'Using seed {seed}, {gpus} gpus and {threads} threads')

    config_path = Path("./params.yaml")

    if not config_path.exists():
        logging.info(f'No config file found at {config_path}. Creating one...')
        config = {
            "language_modeling_train": {
                "cuda_visible_devices": ",".split(gpus),
                "vocab_size": 16000,
                "tokenizer_min_frequency": 2,
                "max_seq_length": 512,
                "mlm_probability": 0.15,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "hidden_size": 768,
                "hidden_dropout_prob": 0.1,
                "epochs": lm_epochs,
                "per_device_train_batch_size": lm_batch_size,
                "per_device_eval_batch_size": int(lm_batch_size * 1.5),
                "seed": seed,
            },
            "classification_train": {
                "cuda_visible_devices": ",".split(gpus),
                "warmup_steps": 500,
                "num_train_epochs": classifier_epochs,
                "per_device_train_batch_size": classifier_batch_size,
                "per_device_eval_batch_size": int(classifier_batch_size * 1.5),
                "seed": seed,
                "output": "probabilities",
                "num_training_samples": 1000,
            },
            "classification_eval": {
                "epsilon": 1e-2,
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
        config['language_modeling_train']['seed'] = seed
        config['classification_train']['seed'] = seed
        config["language_modeling_train"]["cuda_visible_devices"] = ",".split(gpus)
        config["classification_train"]["cuda_visible_devices"] = ",".split(gpus)
        config["datasetrewrite"]["threads"] = threads
        config["language_modeling_train"]["epochs"] = lm_epochs
        config["language_modeling_train"]["per_device_train_batch_size"] = lm_batch_size
        config["language_modeling_train"]["per_device_eval_batch_size"] = int(lm_batch_size * 1.5)
        config["classification_train"]["num_train_epochs"] = classifier_epochs
        config["classification_train"]["per_device_train_batch_size"] = classifier_batch_size
        config["classification_train"]["per_device_eval_batch_size"] = int(classifier_batch_size * 1.5)

        logging.info(f'Saving config to {config_path.absolute()}')
        with open(config_path, "w") as f:
            yaml.dump(config, f)

    data_path = Path("./data/")
    if not data_path.exists():
        logging.info(f'No data folder found at {data_path}. Creating one...')
        data_path.mkdir(parents=True)
        # copy ./labels.tsv to ./data/
        shutil.copy("./labels.tsv", "./data/")

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
