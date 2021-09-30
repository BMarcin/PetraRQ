import logging
from typing import Dict


def dataset_splitter_by_time(
        dataset: list,
        dev_size: float,
        test_size: float,
        train_size: float,
        parts: int
) -> Dict[str, list]:
    assert dev_size + test_size + train_size == 1.0, 'Dev + Test + Train must be equal to 1.0'

    dev_divided = dev_size / parts
    test_divided = test_size / parts
    train_divided = train_size / parts

    divisions = {
        'dev': [],
        'test': [],
        'train': []
    }

    counter = 0
    for i in range(parts):
        start_pos = counter
        end_pos = round(len(dataset) * train_divided) + start_pos
        divisions['train'] += [dataset[item] for item in range(start_pos, end_pos)]
        counter = end_pos
        logging.debug('#{} Adding to train from: {} to {}'.format(str(i), str(start_pos), str(end_pos)))

        start_pos = counter
        end_pos = round(len(dataset) * test_divided) + start_pos
        divisions['test'] += [dataset[item] for item in range(start_pos, end_pos)]
        counter = end_pos
        logging.debug('#{} Adding to test from: {} to {}'.format(str(i), str(start_pos), str(end_pos)))

        start_pos = counter
        end_pos = round(len(dataset) * dev_divided) + start_pos
        divisions['dev'] += [dataset[item] for item in range(start_pos, end_pos)]
        counter = end_pos
        logging.debug('#{} Adding to dev from: {} to {}'.format(str(i), str(start_pos), str(end_pos)))

    start_pos = counter
    end_pos = len(dataset) - len(divisions['dev']) - len(divisions['test']) - len(divisions['train'])
    if end_pos > 0:
        divisions['train'] += [dataset[item] for item in range(start_pos, len(dataset))]
        logging.debug('Adding missing samples to train from: {} to {}'.format(str(start_pos), str(end_pos)))

    return divisions


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    divisions = dataset_splitter_by_time(list(range(34290)), 0.1, 0.2, 0.7, 4)
    print(len(divisions['dev']), len(divisions['test']), len(divisions['train']))
