import torch
import numpy as np
import torch.nn as nn
from collections import Counter


def coll_fn(batch):
    texts = []
    labels = []
    cur_lens = []

    pad_token = 5
    ccc = Counter()

    for (text, label, curr_len) in batch:
        texts.append(torch.tensor(text).to("cpu"))
        labels.append(label)
        cur_lens.append(curr_len)
        ccc[curr_len] += 1

    if sum([0 if x < 0 else 1 for x in cur_lens]) > 0:
        ins = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_token)[:, :max(ccc.keys())]
    else:
        ins = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=pad_token)
    seq_len = ins.shape[1]

    rest = seq_len % 512
    if rest != 0:
        fill_matrix = torch.zeros((ins.shape[0], 512 - rest)).fill_(pad_token).long()
        ins = torch.cat((ins, fill_matrix), dim=1)

    return ins.numpy(), np.array(labels), (~(ins == 5)).long().numpy()
