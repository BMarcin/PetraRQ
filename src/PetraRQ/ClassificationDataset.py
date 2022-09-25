import torch.utils.data
import random
import numpy as np


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_texts, input_labels, unique_labels, tokenizer, increase_each=0.25, start_len=512,
                 no_limit=False, max_len=-1):
        self.input_texts = list(input_texts[0])
        self.input_labels = input_labels
        self.unique_labels = unique_labels
        self.tokenizer = tokenizer
        self.label2idx = {}
        self.no_limit = no_limit
        self.max_len = max_len

        for label in self.unique_labels:
            self.label2idx[label] = len(self.label2idx)

        print(self.label2idx)

        self.curr_len = start_len
        self.increase_groups = []
        for i in range(0, len(self.input_texts), int(len(self.input_texts) * increase_each) + 1):
            self.increase_groups.append((i, i + int(len(self.input_texts) * increase_each)))

        self.random_ids = list(range(len(self.input_texts)))
        # random.shuffle(self.random_ids)
        self.group = 0

        self.increase_counter = 0
        self.increase_after = 1

    def __len__(self):
        return len(self.input_texts)

    def labels2tensor(self, labels):
        return set([self.label2idx[label.strip()] for label in labels])

    def tensor2labels(self, tensor):
        labels = []
        for idx, label in enumerate(self.unique_labels):
            if tensor[idx] == 1:
                labels.append(label)
        return labels

    def __getitem__(self, in_id):
        idx = self.random_ids[in_id]

        tokenized = self.tokenizer.encode(self.input_texts[idx])
        labels = self.labels2tensor(self.input_labels.iloc[idx][0].split(' '))
        # print(self.input_texts[idx][:300])

        length = None

        if not self.no_limit:
            for group_id, (index_start, index_end) in enumerate(self.increase_groups):
                if index_start <= in_id <= index_end:
                    current_group = group_id
                    break

            if current_group != self.group:
                self.group = current_group
                self.curr_len += 512

            length = min([len(tokenized), self.curr_len])

            if 0 < self.max_len < length:
                length = self.max_len
        else:
            self.curr_len = 65536

        if length is not None:
            final_len = length
        else:
            final_len = self.curr_len

        return np.array(tokenized), torch.zeros([len(self.label2idx)]).index_fill_(0, torch.tensor(list(labels)),
                                                                                   1).numpy(), final_len
