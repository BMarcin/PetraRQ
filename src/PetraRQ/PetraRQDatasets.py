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


class LMDS(Dataset):
    def __init__(
            self,
            texts: List[str],
            tokenizer: Tokenizer,
            max_len: int = 512,
            masked_token_probability: float = 0.8,
            other_token_probability: float = 0.1,
            unchanged_token_probability: float = 0.1,
            masked_tokens: float = 0.10,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.masked_token_probability = masked_token_probability
        self.other_token_probability = other_token_probability
        self.unchanged_token_probability = unchanged_token_probability
        self.masked_tokens = masked_tokens

        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")

        self.tokenized_texts = []
        # for text in tqdm(self.tokenizer.encode_batch(self.texts), desc="Encoding texts"):
        #     self.tokenized_texts.append(text.ids)

        logging.info("Saving training data as NumPy array")
        # self.tokenized_texts = np.array(self.tokenized_texts)

        logging.info("Deleting original texts")
        # del self.texts

    def __len__(self):
        # return len(self.texts)
        return len(self.texts) * 5

    def __getitem__(self, index):
        # tokenized_text = np.array(self.tokenizer.encode(self.texts[index]).ids[:self.max_len])
        tokenized_text = np.array(self.tokenizer.encode(self.texts[int(index / 5)]).ids[:self.max_len])
        non_zero_np = (tokenized_text == self.pad_token_id).nonzero()
        if len(non_zero_np) > 0 and non_zero_np[0].size > 0:
            item_padding_start_index = non_zero_np[0][0]
        else:
            item_padding_start_index = len(tokenized_text)

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
            mask = np.ones(output_vector.shape, dtype=bool)
            mask[masked_tokens_ids] = False
            output_vector[mask] = -100
            # output_vector[~np.array(masked_tokens_ids)] = -100

        return input_vector, output_vector


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
            max_len,
            masked_token_probability=0.8,
            other_token_probability=0.1,
            unchanged_token_probability=0.1,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.dev_data = dev_data
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.masked_token_probability = masked_token_probability
        self.other_token_probability = other_token_probability
        self.token_unchanged_probability = unchanged_token_probability

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

        self.tokenizer.enable_padding(
            length=self.max_len,
            pad_token="[PAD]",
        )
        self.tokenizer.enable_truncation(
            max_length=self.max_len,
        )

        self.dev_dataset = LMDS(
            texts=self.dev_data,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            masked_token_probability=self.masked_token_probability,
            other_token_probability=self.other_token_probability,
            unchanged_token_probability=self.token_unchanged_probability
        )
        logging.info(f"LMDS dev_dataset: {len(self.dev_dataset)}")

        self.test_dataset = LMDS(
            texts=self.test_data,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            masked_token_probability=1,
            other_token_probability=0,
            unchanged_token_probability=0
        )
        logging.info(f"LMDS test_dataset: {len(self.test_dataset)}")

        self.train_dataset = LMDS(
            texts=self.train_data,
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            masked_token_probability=1,
            other_token_probability=0,
            unchanged_token_probability=0
        )
        logging.info(f"LMDS train_dataset: {len(self.train_dataset)}")

    def train_tokenizer(self):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            show_progress=True,
            special_tokens=["[MASK]", "[UNK]", "[PAD]"],
        )
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.train_from_iterator(self.train_data, trainer=trainer)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
        level=logging.DEBUG
    )

    # with open("../../data/train/lm.txt", "r", encoding="utf-8") as f:
    #     train_data = f.readlines()
    #
    # with open("../../data/test/lm.txt", "r", encoding="utf-8") as f:
    #     test_data = f.readlines()

    with open("../../data/dev/lm.txt", "r", encoding="utf-8") as f:
        dev_data = f.readlines()[:100]

    lm_ds = LanguageModellingDataset(
        train_data=dev_data,
        test_data=dev_data,
        dev_data=dev_data,
        vocab_size=16000,
        max_len=21
    )
    # for i in range(100):
    #     print(lm_ds.dev_dataset.__getitem__(i))

    print(lm_ds.dev_dataset.__getitem__(0))