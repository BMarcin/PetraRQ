from typing import Any

import torch
from torch import nn

import pytorch_lightning as pl
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(preds, labels):
    preds = (preds >= 0.5).astype(int)
    acc = accuracy_score(preds, labels)

    return {
        'accuracy': acc,
        'f1': f1_score(y_true=labels, y_pred=preds, average='weighted'),
        'precision': precision_score(y_true=labels, y_pred=preds, average='weighted'),
        'recall': recall_score(y_true=labels, y_pred=preds, average='weighted')
    }


class PetraRQ(pl.LightningModule):
    def __init__(
            self,
            d_model,
            num_labels,
            seq_length,
            overlapping_part,
            model=None,
            embeddings=None,
            lr=1e-4,
    ):
        super(PetraRQ, self).__init__()
        self.save_hyperparameters(
            'd_model',
            'num_labels',
            'seq_length',
            'overlapping_part',
            ignore=[
                "embeddings",
                "embeds",
                "model",
                "roberta",
                "embeds.position_ids", "embeds.word_embeddings.weight", "embeds.position_embeddings.weight",
                "embeds.token_type_embeddings.weight", "embeds.LayerNorm.weight", "embeds.LayerNorm.bias"
            ]
        )

        self.d_model = d_model
        self.num_labels = num_labels
        self.seq_length = seq_length
        self.overlapping_part = overlapping_part
        self.lr = lr
        self.overlapping_part = overlapping_part
        self.memory_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.between_norm = nn.LayerNorm(d_model)
        self.activate = nn.GELU()
        self.sigm = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.max_curr_len = 512

        self.embeds = embeddings
        self.roberta = model

        self.fc = nn.Linear(2 * d_model, 2 * d_model)
        self.fc_mem = nn.Linear(d_model, d_model)
        self.fc_between = nn.Linear(d_model, 512)
        self.fc_between2 = nn.Linear(512, d_model)
        self.to_logits = nn.Linear(2 * d_model, num_labels)

        if self.roberta is not None:
            for param in self.roberta.parameters():
                param.requires_grad = False

        if self.embeds is not None:
            for param in self.embeds.parameters():
                param.requires_grad = False

    def forward(self, x_in, attention):
        if self.roberta is None or self.embeds is None:
            raise ValueError("Model or embeddings are not defined")

        floating_memory = None
        output_hidden_layers = None

        i = 0
        while (output_hidden_layers is None) or (i < (int(x_in.shape[1] / self.overlapping_part) - 1)):
            part = self.seq_length / self.overlapping_part
            if floating_memory is None:
                x_pos = self.embeds(x_in[:, :self.seq_length].to(self.device))
                att_part = attention[:, :self.seq_length].to(self.device)
            else:
                overlapping_left = int((self.overlapping_part * (i + part - 1)))
                overlapping_right = int((self.overlapping_part * (i + part)))

                toks = x_in[:, overlapping_left:overlapping_right].to(self.device)

                att_part = torch.cat((att_part[:, self.overlapping_part:],
                                      attention[:, overlapping_left:overlapping_right].to(self.device)), dim=1)
                x_pos = self.embeds(toks)
                x_p1 = x[:, self.overlapping_part:, :]
                x_pos = torch.cat((x_p1, x_pos), dim=1)
                x_pos = self.dropout(x_pos)
                del x_p1

            x = self.roberta(inputs_embeds=x_pos.detach(), attention_mask=att_part).last_hidden_state

            x = self.dropout(x)
            x = self.fc_between(x)
            x = self.activate(x)
            x = self.dropout(x)
            x = self.fc_between2(x)
            x = self.activate(x)

            if floating_memory is None:
                floating_memory = self.dropout(x[:, :self.overlapping_part, :])
                floating_memory = self.fc_mem(floating_memory)
                floating_memory = self.activate(floating_memory)
                output_hidden_layers = self.overlapping_part
            else:
                add = self.dropout(x[:, :self.overlapping_part, :])
                add = floating_memory + self.fc_mem(add)
                add = self.activate(add)
                floating_memory = add
                output_hidden_layers += self.overlapping_part
            del x_pos
            i += 1

        out = torch.cat((floating_memory, x[:, self.overlapping_part:, :]), dim=2)
        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.to_logits(out)
        out = self.sigm(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        xs, ys, attention = batch

        print(xs.shape, ys.shape, attention.shape)

        if xs.shape[1] > self.max_curr_len:
            self.max_curr_len = xs.shape[1]

        x = torch.tensor(xs).to("cpu")
        y = torch.tensor(ys).to(self.device)
        attention = torch.tensor(attention).to("cpu")

        x = self.forward(x, attention)

        loss = F.binary_cross_entropy(x, y)

        metrics = compute_metrics(x.detach().cpu().numpy(), y.long().cpu().numpy())

        self.log('train/f1', metrics['f1'], prog_bar=True, batch_size=x.shape[0])
        self.log('train/accuracy', metrics['accuracy'], prog_bar=True, batch_size=x.shape[0])
        self.log('train/precision', metrics['precision'], prog_bar=True, batch_size=x.shape[0])
        self.log('train/recall', metrics['recall'], prog_bar=True, batch_size=x.shape[0])
        self.log('train/len', xs.shape[1], prog_bar=True, batch_size=x.shape[0])

        self.log('train/loss', loss, prog_bar=True, batch_size=x.shape[0])
        return {
            'loss': loss,
            'train_f1': metrics['f1'],
            'train_acc': metrics['accuracy'],
            'train_precision': metrics['precision'],
            'train_recall': metrics['recall']
        }

    def validation_step(self, batch, batch_idx):
        xs, ys, attention = batch
        xs = xs[:, :self.max_curr_len]
        attention = attention[:, :self.max_curr_len]

        x = torch.tensor(xs).to("cpu")
        y = torch.tensor(ys).to(self.device)
        attention = torch.tensor(attention).to("cpu")

        x = self.forward(x, attention)

        loss = F.binary_cross_entropy(x, y)

        metrics = compute_metrics(x.cpu().numpy(), y.long().cpu().numpy())

        self.log('eval/f1', metrics['f1'], prog_bar=True, batch_size=x.shape[0])
        self.log('eval/accuracy', metrics['accuracy'], prog_bar=True, batch_size=x.shape[0])
        self.log('eval/precision', metrics['precision'], prog_bar=True, batch_size=x.shape[0])
        self.log('eval/recall', metrics['recall'], prog_bar=True, batch_size=x.shape[0])
        self.log('eval/len', xs.shape[1], prog_bar=True, batch_size=x.shape[0])

        self.log('eval/loss', loss, prog_bar=True, batch_size=x.shape[0])
        return {
            'val_loss': loss,
            'val_f1': metrics['f1'],
            'val_acc': metrics['accuracy'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall']
        }

    # def save(self):
    #     for hparam_key, hparam_value in self.hparams.items():
    #         print(f'{hparam_key} = {hparam_value}')
