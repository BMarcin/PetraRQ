import torch.utils.data


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_texts, input_labels, unique_labels, tokenizer):
        self.input_texts = input_texts
        self.input_labels = input_labels
        self.unique_labels = unique_labels
        self.tokenizer = tokenizer
        self.label2idx = {}

        for label in self.unique_labels:
            self.label2idx[label] = len(self.label2idx)

        print(self.label2idx)

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
        tokenized = self.tokenizer(str(self.input_texts.iloc[idx][0]), truncation=True, padding="max_length", max_length=512)
        labels = self.labels2tensor(self.input_labels.iloc[idx][0].split(' '))

        item = {
            'input_ids': torch.tensor(tokenized['input_ids']),
            'attention_mask': torch.tensor(tokenized['attention_mask']),
            'labels': torch.zeros([len(self.label2idx)]).index_fill_(0, torch.tensor(list(labels)), 1)
        }

        return item
