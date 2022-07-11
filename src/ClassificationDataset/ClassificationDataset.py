import torch.utils.data
from tqdm.auto import tqdm


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_texts, input_labels, unique_labels, tokenizer):
        self.input_texts = input_texts
        self.input_labels = input_labels
        self.unique_labels = unique_labels
        self.tokenizer = tokenizer
        self.label2idx = {}

        # self.bp = BasicProcessor()

        for label in self.unique_labels:
            self.label2idx[label] = len(self.label2idx)

        print(self.label2idx)

        self.tokenized_texts = []

        # processed_texts = processing_function(
        #     list(self.input_texts[0]),
        #     self.bp,
        #     threads=12
        # )

        for idx in tqdm(range(len(self.input_texts)), desc='Tokenizing texts'):
            tokenized = self.tokenizer(self.input_texts.iloc[idx][0], truncation=True, padding="max_length", max_length=512)
            self.tokenized_texts.append(tokenized)

    def __len__(self):
        return len(self.input_texts)

    def labels2tensor(self, labels):
        return set([self.label2idx[label.strip().lower()] for label in labels])

    def tensor2labels(self, tensor):
        labels = []
        for idx, label in enumerate(self.unique_labels):
            if tensor[idx] == 1:
                labels.append(label)
        return labels

    def __getitem__(self, idx):
        tokenized = self.tokenized_texts[idx]
        labels = self.labels2tensor(self.input_labels.iloc[idx][0].split(' '))

        item = {
            'input_ids': torch.tensor(tokenized['input_ids']),
            'attention_mask': torch.tensor(tokenized['attention_mask']),
            'labels': torch.zeros([len(self.label2idx)]).index_fill_(0, torch.tensor(list(labels)), 1)
        }

        return item
