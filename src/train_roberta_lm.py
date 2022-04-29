import json
import logging

# import wandb
import os
import random

import numpy as np
import torch
import wandb
import yaml
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig, RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import math

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ["WANDB_DISABLED"] = "true"


def encode(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=config['max_seq_length']
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # load config
    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['language_modeling_train']
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in config['cuda_visible_devices']])
    os.environ["WANDB_PROJECT"] = 'PetraRQ'

    # set random state
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # log to wandb
    logging.info("Logging to wandb...")
    wandb.login()

    # set the vocab size
    vocab_size = config['vocab_size']

    # create special tokens array
    special_tokens = [
        '<url>',
        '<email>',
        '<number>',
        '<date>',
    ]

    # setup datasets paths
    dev_ds = "./data/dev/lm.txt"
    test_ds = "./data/test/lm.txt"
    train_ds = "./data/train/lm.txt"

    # set models path
    models_path = "./models/roberta_lm"
    os.makedirs(models_path, exist_ok=True)

    # train tokenizer
    logging.info("Training tokenizer...")
    bpe = ByteLevelBPETokenizer()
    bpe.train(
        files=[train_ds],
        vocab_size=vocab_size,
        min_frequency=config['tokenizer_min_frequency'],
        special_tokens=special_tokens + [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]
    )
    bpe.save_model(models_path)

    # load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(models_path, max_len=config['max_seq_length'], use_fast=True)
    tokenizer.add_special_tokens({
        'additional_special_tokens': special_tokens
    })

    # build dataset
    logging.info("Building datasets...")
    dataset = load_dataset(
        'text',
        data_files={
            'train': [train_ds],
            'test': [test_ds],
            'dev': [dev_ds]
        }
    )

    tokenized_datasets = dataset.map(
        encode,
        batched=True,
        remove_columns=['text'],
        load_from_cache_file=True,
    )

    # build DS collocator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config['mlm_probability']
    )

    # build model
    model_config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=config['max_seq_length'] + 2,
        num_attention_heads=config['num_attention_heads'],
        num_hidden_layers=config['num_hidden_layers'],
        type_vocab_size=1,
        layer_norm_eps=0.00001,
        hidden_size=config['hidden_size'],
        hidden_dropout_prob=config['hidden_dropout_prob']
    )
    model = RobertaForMaskedLM(config=model_config)

    logging.info("Model parameters: {}".format(model.num_parameters()))

    # build trainer
    training_args = TrainingArguments(
        output_dir=models_path,
        overwrite_output_dir=True,
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        per_device_eval_batch_size=config['per_device_eval_batch_size'],
        save_steps=10_000,
        save_total_limit=3,
        do_train=True,
        do_eval=True,
        no_cuda=False,
        logging_steps=2500,
        eval_steps=2500,
        evaluation_strategy='steps',
        report_to="wandb",
        run_name="petrarq-roberta-lm"
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets['dev']
    )

    logging.info("Starting training...")
    trainer.train()

    # evaluate
    logging.info("Starting evaluation...")
    eval_output = trainer.evaluate(tokenized_datasets["test"])
    perplexity = math.exp(eval_output["eval_loss"])

    # save model
    logging.info("Saving model")
    trainer.save_model()

    logging.info("Save scores")
    scores = {
        "perplexity": perplexity,
        "eval_loss": eval_output["eval_loss"]
    }
    logging.info("Model perplexity: {}, loss: {}".format(perplexity, eval_output["eval_loss"]))

    with open("scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    wandb.finish()
