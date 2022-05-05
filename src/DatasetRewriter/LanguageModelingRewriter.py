import logging
from typing import List, Union

from MordinezNLP.processors import BasicProcessor


def processing_function(
        texts_list: Union[List[str], str],
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


def process_datasets(
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

    print('Used special chars')
    print(bp.get_special_tokens())

    return dev_output_texts, test_output_texts, train_output_texts
