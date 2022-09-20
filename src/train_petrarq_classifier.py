import logging
import os
import shutil

import pandas as pd
import wandb
import yaml

from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, RobertaModel

from PetraRQ.ClassificationDataset import ClassificationDataset
from PetraRQ.CollateFunction import coll_fn
from PetraRQ.PetraRQ import PetraRQ

import pytorch_lightning as pl
import warnings

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info("Loading config...")
    config = yaml.safe_load(open("./params.yaml"))['classification_train']
    config_train = yaml.safe_load(open("./params.yaml"))['language_modeling_train']
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(item) for item in config['cuda_visible_devices']])

    seed_everything(config['seed'], workers=True)

    if config['use_wandb_logging']:
        os.environ["WANDB_PROJECT"] = 'PetraRQ-Classifier'
        os.environ["WANDB_DISABLED"] = "false"

        logging.info("Logging to wandb...")
        wandb.login()
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # setup datasets paths
    dev_ds = "./data/dev/"
    train_ds = "./data/train/"

    # set models path
    lm_model_path = "./models/roberta_lm"
    models_path = "./models/petrarq_classifier"
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(os.path.join(models_path, "checkpoints"), exist_ok=True)

    # define special characters
    logging.info("Defining special characters...")
    special_tokens = [
        '<url>',
        '<email>',
        '<number>',
        '<date>',
    ]

    logging.info("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained(lm_model_path, max_len=config_train['max_seq_length'],
                                                     use_fast=True)

    # add special tokens
    tokenizer.add_special_tokens({
        'additional_special_tokens': special_tokens
    })

    # Load the data
    logging.info('Loading data...')
    data_train = pd.read_csv("./data/train/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    data_dev = pd.read_csv("./data/dev/processed.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_train = pd.read_csv("./data/train/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    labels_dev = pd.read_csv("./data/dev/expected.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    unique_labels = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)

    # using config You can adjust how many random samples are used in training
    if config['num_training_samples'] > 0:
        # temporary combine train and labels
        combined = list(zip(data_train[0], labels_train[0]))
        # combined = sorted(combined, key=lambda x: len(x[1].split(" ")), reverse=True)

        # prepare "table" for labels where columns are labels and samples are rows, each sample can have multiple labels
        category_id = dict((label, set()) for label in unique_labels[0].unique())

        # fill the table
        for i, (doc, doc_labels) in enumerate(combined):
            single_doc_labels = doc_labels.split(" ")
            for label in single_doc_labels:
                category_id[label].add((i, len(single_doc_labels)))

        # sort each column by the number of labels in each sample
        for label in category_id.keys():
            category_id[label] = sorted(category_id[label], key=lambda x: x[1], reverse=True)

        # select samples for training
        items_ids = []
        used_ids = set()

        curr_id = 0
        while len(items_ids) < config['num_training_samples']:
            for label in category_id.keys():
                # print(label)
                if len(items_ids) >= config['num_training_samples']:
                    break

                try:
                    doc_id = category_id[label][curr_id][0]
                    if doc_id in used_ids:
                        pass

                    items_ids.append(doc_id)
                    used_ids.add(doc_id)
                except:
                    continue
            curr_id += 1

        data_train = data_train.loc[items_ids]
        labels_train = labels_train.loc[items_ids]

    # Make unique labels
    logging.info('Making unique labels...')
    unique_labels_tsv = pd.read_csv("./data/labels.tsv", delimiter='\t', header=None, encoding="utf8", quoting=0)
    unique_labels = unique_labels_tsv[0].tolist()

    # create datasets
    logging.info("Creating datasets...")
    dev_ds = ClassificationDataset(data_dev, labels_dev, unique_labels, tokenizer, increase_each=config['increase_each'], start_len=2048,
                                   no_limit=True)
    train_ds = ClassificationDataset(data_train, labels_train, unique_labels, tokenizer, no_limit=False,
                                     increase_each=config['increase_each'], start_len=2048)
    num_labels = len(unique_labels)

    train_data_loader = DataLoader(
        train_ds,
        batch_size=config['train_batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=coll_fn,
    )

    dev_data_loader = DataLoader(
        dev_ds,
        batch_size=config['dev_batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=coll_fn
    )

    # define model
    logging.info("Defining model...")
    model = RobertaModel.from_pretrained(lm_model_path)
    embeds = model.embeddings

    petra = PetraRQ(
        d_model=config_train['hidden_size'],
        num_labels=len(unique_labels),
        seq_length=config['seq_length'],
        overlapping_part=config['overlapping_part'],
        # steps=config['steps'],
        embeddings=embeds,
        model=model,
        lr=float(config['lr']),
    )

    # build trainer
    logging.info("Building trainer...")
    wb_logger = {}
    if config['use_wandb_logging']:
        wandb_logger = WandbLogger(
            project="PetraRQ-Classifier",
            name="PetraRQ-1k",
            log_model="all"
        )

        wandb_logger.experiment.config['batch_size'] = config['train_batch_size']
        wandb_logger.experiment.config['epochs'] = config['epochs']
        wandb_logger.experiment.config['overlapping_part'] = config['overlapping_part']
        wandb_logger.experiment.config['lr'] = config['lr']
        wandb_logger.experiment.config['num_training_samples'] = config['num_training_samples']

        wb_logger = {'logger': wandb_logger}

    checkpoint_callback = ModelCheckpoint(
        dirpath='./models/petrarq_classifier/checkpoints',
        save_top_k=3,
        monitor='eval/f1',
        mode='max',
        filename='petrarq-{epoch}-{eval/f1:.2f}',
        auto_insert_metric_name=False
    )

    trainer = pl.Trainer(
        devices=[int(item) for item in config['cuda_visible_devices']],
        # max_steps=config['steps'],
        max_epochs=config['epochs'],
        log_every_n_steps=2,
        accelerator='gpu',
        val_check_interval=0.3,
        default_root_dir=models_path,
        enable_checkpointing=True,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(
                monitor='eval/f1',
                mode='max',
                patience=3,
                check_finite=True,
            ),
            LearningRateMonitor(logging_interval='step'),
            TQDMProgressBar(refresh_rate=1),
        ],
        **wb_logger,
        reload_dataloaders_every_n_epochs=0,
    )

    warnings.filterwarnings('ignore')

    # train model
    logging.info("Training model...")
    trainer.fit(petra, train_data_loader, dev_data_loader)

    # save model
    logging.info("Saving model...")
    if os.path.exists(checkpoint_callback.best_model_path):
        shutil.copyfile(checkpoint_callback.best_model_path, os.path.join(models_path, 'pytorch_model.ckpt'))
    else:
        trainer.save_checkpoint(
            os.path.join(models_path, 'pytorch_model.ckpt'),
        )
