import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import logging

from src.PetraRQ.PetraRQ import PetraRQ
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.PetraRQ.PetraRQDatasets import LanguageModellingDataset

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        level=logging.INFO
    )

    torch.manual_seed(1)

    with open("../../data/train/lm.txt", "r", encoding="utf-8") as f:
        train_data = f.readlines()

    # with open("../../data/test/lm.txt", "r", encoding="utf-8") as f:
    #     test_data = f.readlines()

    with open("../../data/dev/lm.txt", "r", encoding="utf-8") as f:
        dev_data = f.readlines()

    lm_ds = LanguageModellingDataset(
        train_data=train_data,
        test_data=dev_data,
        dev_data=dev_data,
        vocab_size=16000,
        max_len=512
    )

    train_data_loader = DataLoader(
        lm_ds.train_dataset,
        batch_size=28,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    dev_data_loader = DataLoader(
        lm_ds.dev_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    steps = 40000

    petra = PetraRQ(
        d_model=512,
        num_tokens=16000,
        seq_length=512,
        depth=12,
        k=768,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=True,
        dropout=0.1,
        steps=steps,
    )

    wandb_logger = WandbLogger(
        project="PetraRQ",
        name="PetraRQ_v1_dev",
        log_model="all"
    )

    trainer = pl.Trainer(
        devices=1,
        max_steps=steps,
        log_every_n_steps=10,
        accelerator='gpu',
        accumulate_grad_batches=10,
        val_check_interval=1.0,
        # val_check_interval=300,
        default_root_dir='./PetraRQmodel',
        enable_checkpointing=False,
        callbacks=[
            # ModelCheckpoint(
            #     dirpath='./PetraRQmodel/checkpoints',
            #     save_top_k=3,
            #     monitor='eval/loss',
            #     mode='min',
            #     filename='petrarq-{epoch}-{val_loss:.2f}.ckpt'
            # ),
            EarlyStopping(
                monitor='eval/loss',
                mode='min',
                patience=10,
                check_finite=True,
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        logger=wandb_logger
    )
    wandb_logger.watch(petra, log='all')

    wandb_logger.experiment.config['batch_size'] = 28
    wandb_logger.experiment.config['steps'] = steps
    wandb_logger.experiment.config['d_model'] = 512
    wandb_logger.experiment.config['num_tokens'] = 16000
    wandb_logger.experiment.config['seq_length'] = 512
    wandb_logger.experiment.config['depth'] = 10
    wandb_logger.experiment.config['k'] = 768
    wandb_logger.experiment.config['heads'] = 8
    wandb_logger.experiment.config['dim_head'] = None
    wandb_logger.experiment.config['one_kv_head'] = False
    wandb_logger.experiment.config['share_kv'] = True
    wandb_logger.experiment.config['dropout'] = 0.1
    wandb_logger.experiment.config['train_file'] = '../../data/dev/train.txt'
    # wandb_logger.experiment.config['test_file'] = '../../data/dev/lm.txt'
    wandb_logger.experiment.config['dev_file'] = '../../data/dev/lm.txt'
    wandb_logger.experiment.config['optimizer'] = 'adagrad'

    trainer.fit(petra, train_data_loader, dev_data_loader)
