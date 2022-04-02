import argparse
import logging
import os

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
        exit(parser.print_usage())

    # check if threads is set
    if not threads:
        exit(parser.print_usage())

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


