import logging
from typing import List

from MordinezNLP.processors import BasicProcessor
from tqdm.auto import tqdm


def processing_function(
        texts_list: List[str],
        bp: BasicProcessor,
        threads: int = 12,
):
    return bp.process(
        texts_list,
        language='en',
        fix_unicode=True,
        lower=False,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=False,
        no_numbers=True,
        no_digits=True,
        no_currency_symbols=False,
        no_punct=False,
        no_math=False,
        no_dates=True,
        no_lists=False,
        no_brackets=False,
        no_multiple_chars=False,
        use_pos_tagging=False,
        list_processing_threads=threads,
        tokenizer_threads=threads
    )


def rewrite_datasets_to_txt(
        dev_input_texts: List[str],
        test_input_texts: List[str],
        train_input_texts: List[str],
        threads: int = 12,
):
    bp = BasicProcessor()

    logging.info('Processing dev texts')
    dev_output_texts = processing_function(dev_input_texts, bp, threads)

    logging.info('Processing test texts')
    test_output_texts = processing_function(test_input_texts, bp, threads)

    logging.info('Processing train texts')
    train_output_texts = processing_function(train_input_texts, bp, threads)

    with open("../data/dev/lm.txt", "w", encoding="utf8") as f:
        for item in tqdm(dev_output_texts, desc="Saving dev ds"):
            f.write(item + '\n')

    with open("../data/test/lm.txt", "w", encoding="utf8") as f:
        for item in tqdm(test_output_texts, desc="Saving test ds"):
            f.write(item + '\n')

    with open("../data/train/lm.txt", "w", encoding="utf8") as f:
        for item in tqdm(train_output_texts, desc="Saving train ds"):
            f.write(item + '\n')

    print('Used special chars')
    print(bp.get_special_tokens())
