import hashlib
import logging
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tqdm.auto import tqdm

import copy

from transformers import DataCollatorForLanguageModeling, BertTokenizer, AutoTokenizer


class LMDS(Dataset):
    def __init__(
            self,
            texts: List[str],
            tokenizer: Tokenizer,
            # max_len: int = 512,
            batch_size: int = 32,
            masked_token_probability: float = 0.8,
            other_token_probability: float = 0.1,
            unchanged_token_probability: float = 0.1,
            masked_tokens: float = 0.10,
            duplicate_dataset_ratio: int = 1,
            use_incremental_samples: bool = False,
            incremental_samples_min: int = 10,
            incremental_samples_step: int = 20,
            increment_every_x_steps: int = -1
    ):
        assert duplicate_dataset_ratio >= 1, "duplicate_dataset_ratio must be >= 1"

        self.texts = texts
        # random.shuffle(self.texts)
        self.texts = sorted(self.texts, key=lambda x: len(x))

        self.tokenizer = tokenizer
        # self.batch_size = batch_size
        # self.max_len = max_len

        self.masked_token_probability = masked_token_probability
        self.other_token_probability = other_token_probability
        self.unchanged_token_probability = unchanged_token_probability
        self.masked_tokens = masked_tokens
        self.duplicate_dataset_ratio = duplicate_dataset_ratio
        self.use_incremental_samples = use_incremental_samples
        self.epochs_done = 0
        self.total_items_get = 0
        self.incremental_samples_min = incremental_samples_min
        self.incremental_samples_step = incremental_samples_step
        self.increment_every_x_steps = increment_every_x_steps

        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")

        # self.pad_lengths = []
        # for i in range(0, len(self.texts), self.batch_size):
        #     batch_pad_length = max([len(x) for x in self.texts[i:i + self.batch_size if i + self.batch_size < len(self.texts) else len(self.texts)]])
        #     # print(i, i + self.batch_size, batch_pad_length)
        #     self.pad_lengths = self.pad_lengths + [batch_pad_length for _ in range(self.batch_size)]

        # self.tokenized_texts = []
        # for text in tqdm(self.tokenizer.encode_batch(self.texts), desc="Encoding texts"):
        #     self.tokenized_texts.append(text.ids)

        # logging.info("Saving training data as NumPy array")
        # self.tokenized_texts = np.array(self.tokenized_texts)

        # logging.info("Deleting original texts")
        # del self.texts

    def __len__(self):
        # return len(self.texts)
        return len(self.texts) * self.duplicate_dataset_ratio

    def __getitem__(self, index):
        # tokenized_text = np.array(self.tokenizer.encode(self.texts[index]).ids[:self.max_len])
        # batch_pad_length = self.pad_lengths[index]
        # print('batch_pad_length', batch_pad_length)
        tokenized_text = np.array(self.tokenizer.encode(self.texts[int(index / self.duplicate_dataset_ratio)]).ids)
        # if len(tokenized_text) < batch_pad_length:
        #     tokenized_text = np.concatenate([tokenized_text, np.array([self.pad_token_id for _ in range(batch_pad_length - len(tokenized_text))])])

        non_zero_np = (tokenized_text == self.pad_token_id).nonzero()
        if len(non_zero_np) > 0 and non_zero_np[0].size > 0:
            item_padding_start_index = non_zero_np[0][0]
        else:
            item_padding_start_index = len(tokenized_text)

        incremental_index = self.incremental_samples_min + self.epochs_done * self.incremental_samples_step
        if self.use_incremental_samples and item_padding_start_index >= self.incremental_samples_min and incremental_index <= item_padding_start_index:
            logging.debug("Using incremental samples")
            if self.epochs_done == 0:
                logging.debug("Epoch 0")
                item_padding_start_index = self.incremental_samples_min
            else:
                logging.debug("Epoch > 0")
                item_padding_start_index = incremental_index

        logging.debug(f"Padding start index: {item_padding_start_index}")

        input_vector = copy.deepcopy(tokenized_text)
        output_vector = copy.deepcopy(tokenized_text)

        masked_tokens_ids = []

        for i, in_id in enumerate(input_vector):
            if i >= item_padding_start_index:
                logging.debug(f"i: {i}")
                break

            if random.random() < self.masked_tokens:
                random_mode = np.random.choice(
                    ["mask", "other", "unchanged"],
                    p=[self.masked_token_probability, self.other_token_probability, self.unchanged_token_probability]
                )

                if random_mode == "mask":
                    logging.debug(f"Masking token {in_id}")
                    input_vector[i] = self.mask_token_id
                    masked_tokens_ids.append(i)
                elif random_mode == "other":
                    logging.debug(f"Changing token {in_id} to random token")
                    random_token_id = np.random.randint(self.tokenizer.get_vocab_size())
                    input_vector[i] = random_token_id
                    masked_tokens_ids.append(i)
                elif random_mode == "unchanged":
                    logging.debug(f"Keeping token {in_id}")
                    masked_tokens_ids.append(i)

        logging.debug(f"Masked tokens: {masked_tokens_ids}")
        logging.debug(f"Input vector: {input_vector}")
        logging.debug(f"Output vector: {output_vector}")
        if len(masked_tokens_ids) > 0:
            logging.debug(f"Masking vector with -100")
            mask = np.ones(output_vector.shape, dtype=bool)
            mask[masked_tokens_ids] = False
            output_vector[mask] = -100
            # output_vector[~np.array(masked_tokens_ids)] = -100
        else:
            logging.debug(f"No masked tokens")
            output_vector = -100 * np.ones(output_vector.shape)

        if self.increment_every_x_steps >= 1:
            # print(index)
            if self.total_items_get % self.increment_every_x_steps == 0 and self.total_items_get > 0:
                self.epochs_done += 1
                logging.info(f"Incrementing epochs done to {self.epochs_done}")
                print(f"Incrementing epochs done to {self.epochs_done}")
        else:
            if self.total_items_get == len(self) - 1:
                self.epochs_done += 1
                logging.info(f"Incrementing epochs done to {self.epochs_done}")
                print(f"Incrementing epochs done to {self.epochs_done}")

        if self.total_items_get == len(self) - 1:
            self.total_items_get = 0
        self.total_items_get += 1

        if self.use_incremental_samples:
            return input_vector.astype(np.int64), output_vector.astype(np.int64), np.array([item_padding_start_index]).astype(np.int64)
        else:
            return input_vector.astype(np.int64), output_vector.astype(np.int64)


    # def __getitem__(self, idx):
    #     random_mode = np.random.choice(
    #         ["mask", "other", "unchanged"],
    #         p=[self.masked_token_probability, self.other_token_probability, self.unchanged_token_probability]
    #     )
    #
    #     logging.debug(f"LMDS random_mode: {random_mode}")
    #
    #     # tokenized_text = self.tokenizer.encode(self.texts[idx]).ids
    #     tokenized_text = self.tokenized_texts[idx]
    #     tokenized_text = tokenized_text[:self.max_len]
    #     logging.debug(f"LMDS tokenized text length: {len(tokenized_text)}")
    #
    #     non_zero_np = (tokenized_text == self.pad_token_id).nonzero()
    #     if len(non_zero_np) > 0 and non_zero_np[0].size > 0:
    #         item_padding_start_index = non_zero_np[0][0]
    #     else:
    #         item_padding_start_index = len(tokenized_text)
    #
    #     random_token_mask = random.randint(0, item_padding_start_index - 1)
    #     logging.debug(f"LMDS random_token_mask: {random_token_mask}")
    #
    #     if random_mode == "mask":
    #         input_text = copy.deepcopy(tokenized_text)
    #         output = np.random.rand(tokenized_text.shape[0])
    #         output.fill(-100)
    #
    #         output[random_token_mask] = input_text[random_token_mask]
    #         input_text[random_token_mask] = self.mask_token_id
    #     elif random_mode == "other":
    #         input_text = copy.deepcopy(tokenized_text)
    #
    #         output_token = random.randint(0, self.tokenizer.get_vocab_size())
    #
    #         output = np.random.rand(tokenized_text.shape[0])
    #         output.fill(-100)
    #
    #         input_text[random_token_mask] = self.mask_token_id
    #         output[random_token_mask] = output_token
    #     else:
    #         input_text = copy.deepcopy(tokenized_text)
    #
    #         output = np.random.rand(tokenized_text.shape[0])
    #         output.fill(-100)
    #
    #         output[random_token_mask] = input_text[random_token_mask]
    #
    #     return torch.tensor(input_text).long(), \
    #            torch.tensor(output).long(), \
    #            torch.tensor(random_token_mask).long()


class LanguageModellingDataset:
    def __init__(
            self,
            train_data,
            test_data,
            dev_data,
            vocab_size,
            # batch_size,
            # max_len,
            masked_token_probability=0.8,
            other_token_probability=0.1,
            unchanged_token_probability=0.1,
            masking_token_parts=0.1,
            duplicate_dataset_ratio=1,
            use_incremental_samples: bool = False,
            incremental_samples_min: int = 10,
            incremental_samples_step: int = 20,
            increment_every_x_steps: int = -1
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = dev_data
        self.vocab_size = vocab_size
        # self.max_len = max_len
        # self.batch_size = batch_size

        self.masked_token_probability = masked_token_probability
        self.other_token_probability = other_token_probability
        self.token_unchanged_probability = unchanged_token_probability
        self.masking_token_parts = masking_token_parts
        self.duplicate_dataset_ratio = duplicate_dataset_ratio
        self.use_incremental_samples = use_incremental_samples
        self.incremental_samples_min = incremental_samples_min
        self.incremental_samples_step = incremental_samples_step
        self.increment_every_x_steps = increment_every_x_steps

        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        data_hash = hashlib.new("md5")
        for item in tqdm(train_data, desc="Hash training data"):
            data_hash.update(str(item).encode("utf-8"))

        logging.info(f"Training data hash: {data_hash.hexdigest()}")
        if os.path.exists("./tokenizer_{}.json".format(data_hash.hexdigest())):
            logging.info("Loading tokenizer from disk")
            self.tokenizer = Tokenizer.from_file("./tokenizer_{}.json".format(data_hash.hexdigest()))
        else:
            logging.info("Training tokenizer")
            self.train_tokenizer()
            logging.info("Saving tokenizer to disk")
            self.tokenizer.save("./tokenizer_{}.json".format(data_hash.hexdigest()))
            logging.info("Tokenizer saved to: ./tokenizer_{}.json".format(data_hash.hexdigest()))

        # self.tokenizer.enable_padding(
        #     length=self.max_len,
        #     pad_token="[PAD]",
        # )
        # self.tokenizer.enable_truncation(
        #     max_length=self.max_len,
        # )

        self.dev_dataset = LMDS(
            texts=self.dev_data,
            tokenizer=self.tokenizer,
            # max_len=self.max_len,
            # batch_size=self.batch_size,
            masked_token_probability=self.masked_token_probability,
            other_token_probability=self.other_token_probability,
            unchanged_token_probability=self.token_unchanged_probability,
            masked_tokens=self.masking_token_parts,
            duplicate_dataset_ratio=1,
            use_incremental_samples=False,
            incremental_samples_min=self.incremental_samples_min,
            incremental_samples_step=self.incremental_samples_step,
            increment_every_x_steps=self.increment_every_x_steps
        )
        logging.info(f"LMDS dev_dataset: {len(self.dev_dataset)}")

        if self.test_data is not None:
            self.test_dataset = LMDS(
                texts=self.test_data,
                tokenizer=self.tokenizer,
                # max_len=self.max_len,
                # batch_size=self.batch_size,
                masked_token_probability=1,
                other_token_probability=0,
                unchanged_token_probability=0,
                masked_tokens=self.masking_token_parts,
                duplicate_dataset_ratio=self.duplicate_dataset_ratio,
                use_incremental_samples=self.use_incremental_samples,
                incremental_samples_min=self.incremental_samples_min,
                incremental_samples_step=self.incremental_samples_step,
                increment_every_x_steps=self.increment_every_x_steps
            )
            logging.info(f"LMDS test_dataset: {len(self.test_dataset)}")

        self.train_dataset = LMDS(
            texts=self.train_data,
            tokenizer=self.tokenizer,
            # max_len=self.max_len,
            # batch_size=self.batch_size,
            masked_token_probability=1,
            other_token_probability=0,
            unchanged_token_probability=0,
            masked_tokens=self.masking_token_parts,
            duplicate_dataset_ratio=self.duplicate_dataset_ratio,
            use_incremental_samples=self.use_incremental_samples,
            incremental_samples_min=self.incremental_samples_min,
            incremental_samples_step=self.incremental_samples_step,
            increment_every_x_steps=self.increment_every_x_steps
        )
        logging.info(f"LMDS train_dataset: {len(self.train_dataset)}")

    def train_tokenizer(self):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            show_progress=True,
            special_tokens=["[MASK]", "[UNK]", "[PAD]"],
            min_frequency=3

        )
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train_from_iterator(self.train_data, trainer=trainer)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        level=logging.INFO
    )

    # with open("../../data/train/lm.txt", "r", encoding="utf-8") as f:
    #     train_data = f.readlines()
    #
    # with open("../../data/test/lm.txt", "r", encoding="utf-8") as f:
    #     test_data = f.readlines()

    with open("../../data/dev/lm.txt", "r", encoding="utf-8") as f:
        dev_data = f.readlines()[:1000]

    lm_ds = LanguageModellingDataset(
        train_data=dev_data,
        test_data=dev_data,
        dev_data=dev_data,
        vocab_size=16000,
        # batch_size=7,
        # max_len=512,
        masking_token_parts=0.15,
        use_incremental_samples=False,
        incremental_samples_min=5,
        incremental_samples_step=5,
        increment_every_x_steps=20,
    )

    # print(lm_ds.train_dataset.pad_lengths)

    for _ in range(2):
        # for i in range(30):
        for i in range(len(lm_ds.dev_dataset)):
            item = lm_ds.dev_dataset.__getitem__(len(lm_ds.dev_dataset) - i - 1)
            print(i, item[0].shape, item[1].shape)
            if i >= 15:
                break
        break

    # print(lm_ds.dev_dataset.__getitem__(0))
    # print(lm_ds.dev_dataset.__getitem__(len(lm_ds.dev_dataset) - 1))
    # print(lm_ds.dev_dataset.__getitem__(0))
    # print(lm_ds.dev_dataset.__getitem__(len(lm_ds.dev_dataset) - 1))
    # print(lm_ds.dev_dataset.__getitem__(0))
    # print(lm_ds.dev_dataset.__getitem__(len(lm_ds.dev_dataset) - 1))
    # print(lm_ds.dev_dataset.__getitem__(0))
    # print(lm_ds.dev_dataset.__getitem__(len(lm_ds.dev_dataset) - 1))
    # tknzr = AutoTokenizer.from_pretrained("bert-base-uncased")
    #
    # collocator = DataCollatorForLanguageModeling(
    #    tknzr,
    # )
    # print(collocator.torch_mask_tokens(
    #     torch.tensor([
    #         tknzr.encode("Siema siemanko"),
    #         tknzr.encode("Siema siemanko"),
    #         tknzr.encode("Siema siemanko"),
    #         tknzr.encode("Siema siemanko"),
    #         tknzr.encode("Siema siemanko"),
    #         tknzr.encode("Siema siemanko"),
    #         tknzr.encode("Siema siemanko")
    #     ]),
    # ))
