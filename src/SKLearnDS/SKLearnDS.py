import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class SKlearnDS:
    def __init__(self, data, labels, label2idx):
        self.label2idx = label2idx

        # print(len(labels), len(data))

        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(list(data))

        self.tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

        self.X_train_labels = []

        if labels is not None:
            for index, label in labels.iterrows():
                label = str(label[0]).split(' ')
                label = set([self.label2idx[s_label.strip().lower()] for s_label in label])
                label = torch.zeros([len(self.label2idx)]).index_fill_(0, torch.tensor(list(label)), 1).numpy()

                self.X_train_labels.append(label)
            self.X_train_labels = np.array(self.X_train_labels)

    def get(self):
        return self.X_train_tfidf, self.X_train_labels

    @staticmethod
    def tensor2labels(tensor, unique_labels):
        labels = []
        for idx, label in enumerate(unique_labels):
            if tensor[idx] == 1:
                labels.append(label)
        return labels
