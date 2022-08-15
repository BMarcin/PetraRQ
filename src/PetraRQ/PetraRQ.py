import logging
import math
from operator import itemgetter

import torch
import wandb
from linformer.linformer import GELU, Linformer
from linformer.reversible import ReversibleSequence
from torch import nn

import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR


# from src.PetraRQ.PetraRQReversible import ReversibleSequence


def tensor_std_initializer(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x)


class NormOverLayer(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x): # obsluga positional encodings
        x = self.norm(x)
        return self.fn(x)


class RelativeLogitPositionalEncoding(nn.Module):
    def __init__(self, d_model, num_embeddings, seq_length):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length

        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=d_model,
            # padding_idx=0,
            _weight=tensor_std_initializer(torch.zeros(num_embeddings, d_model)),
        )

        self.position_encodings = nn.Embedding(
            num_embeddings=seq_length,
            embedding_dim=d_model,
            _weight=tensor_std_initializer(torch.zeros(seq_length, d_model)),
        )

        # self.position_encodings = nn.Parameter(tensor_std_initializer(torch.zeros(seq_length)))

    # def forward(self, x, positions):
    #     return self.embeddings(x), self.position_encodings[positions]
    def forward(self, x, **kwargs):
        # return self.embeddings(x), self.position_encodings
        return self.embeddings(x) + self.position_encodings(torch.arange(x.shape[1]).to(x.device))


class PetraRQSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = dim_head if dim_head is not None else dim // heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(tensor_std_initializer(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(tensor_std_initializer(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self, x, context=None):
        b, n, d, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k
        # if 'position_encodings' in kwargs:
        #     print('using positional encodings')
        #     position_encodings = kwargs['position_encodings']
        # else:
        #     position_encodings = None

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(x)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention
        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        # if positional_encodings is not None:
        #     print('using positional encodings')
        #     attn = torch.einsum('bhtn,t->bhtn', attn, positional_encodings)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class PetraRQFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., glu=False):
        super().__init__()
        activation = GELU

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class PetraRQ(pl.LightningModule):
    def __init__(
            self,
            d_model,
            num_tokens,
            seq_length,
            depth,
            k=256,
            heads=8,
            dim_head=None,
            one_kv_head=False,
            share_kv=True,
            dropout=0.1,
            steps=1000,
            lr_min=5e-5,
            lr_max=1e-3,
            optim="adagrad"
    ):
        super(PetraRQ, self).__init__()

        self.d_model = d_model
        self.num_tokens = num_tokens
        self.seq_length = seq_length
        self.depth = depth
        self.k = k
        self.heads = heads
        self.dim_head = dim_head
        self.one_kv_head = one_kv_head
        self.share_kv = share_kv
        self.dropout = dropout
        self.steps = steps
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.optim = optim
        self.activation = GELU()
        self.out_norm = nn.LayerNorm(num_tokens)

        assert (self.optim == 'adam' or self.optim == 'adagrad'), 'Optim must be set to "adam" or "adagrad"'

        self.token_emb = nn.Embedding(num_tokens, d_model)
        self.pos_emb = nn.Embedding(seq_length, d_model)
        self.linformer = Linformer(d_model, seq_length, depth, k=k, heads=heads, dim_head=dim_head,
                                   one_kv_head=one_kv_head, share_kv=share_kv, reversible=True, dropout=dropout)
        self.to_logits = nn.Linear(d_model, num_tokens)

        # self.embeddings = RelativeLogitPositionalEncoding(
        #     d_model=d_model,
        #     num_embeddings=num_tokens,
        #     seq_length=seq_length,
        # )
        #
        # self.petrarq_layers = nn.ModuleList()
        # for _ in range(depth):
        #     attention = PetraRQSelfAttention(
        #         d_model,
        #         seq_len=seq_length,
        #         k=k,
        #         heads=heads,
        #         dim_head=dim_head,
        #         one_kv_head=one_kv_head,
        #         share_kv=share_kv,
        #         dropout=dropout,
        #     )
        #
        #     ff = PetraRQFeedForward(
        #         d_model,
        #         dropout=dropout,
        #         glu=True,
        #     )
        #
        #     self.petrarq_layers.append(
        #         nn.ModuleList([
        #             NormOverLayer(d_model, attention),
        #             NormOverLayer(d_model, ff),
        #         ])
        #     )
        #
        #     self.net = ReversibleSequence(self.petrarq_layers)
        #     self.outs = nn.Linear(d_model, num_tokens)

    def forward(self, x, **kwargs):
        # x, position_encodings = self.embeddings(x)
        # x = self.embeddings(x)
        # x = self.net(x)
        # x = self.outs(x)
        # x = self.activation(x)
        # x = self.out_norm(x)
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.linformer(x)
        out = self.to_logits(x)
        return out

    def configure_optimizers(self):
        if self.optim == 'adagrad':
            optimizer = torch.optim.Adagrad(
                self.parameters(),
                lr=self.lr_min,
                weight_decay=0.01
            )
        elif self.optim == 'adam':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr_min,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )

        # lr_scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr_max,
        #     total_steps=self.steps,
        #     cycle_momentum=False,
        # )

        # return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
        return optimizer

    def training_step(self, batch, batch_idx):
        # x, y, token_pos = batch
        x, y = batch
        x = self.forward(x)
        # print('x shape', x.shape)
        # print('y shape', y.shape)
        # print(token_pos)
        # print('output shape', x.shape)
        # print('target shape', y.shape)
        # print('cross entropy x', x.view(-1, self.num_tokens).shape)
        # print('cross entropy y', y.long().view(-1).shape)

        # loss = F.cross_entropy(x[range(x.shape[0]), token_pos, :], y)
        loss = F.cross_entropy(x.view(-1, self.num_tokens), y.long().view(-1))
        self.log('train/loss', loss, prog_bar=True)
        wandb.log({'train/loss': loss})

        perplexity = torch.exp(loss)
        self.log('train/perplexity', perplexity, prog_bar=True)
        wandb.log({'train/perplexity': perplexity})

        # wandb.log({'train/learning_rate': self.optimizers[0].param_groups[0]['lr']})

        return {'loss': loss, 'perplexity': perplexity}

    def validation_step(self, batch, batch_idx):
        # x, y, token_pos = batch
        x, y = batch
        x = self.forward(x)
        # loss = F.cross_entropy(x[range(x.shape[0]), token_pos, :], y)
        loss = F.cross_entropy(x.view(-1, self.num_tokens), y.long().view(-1))
        perplexity = torch.exp(loss)
        self.log('eval/loss', loss, prog_bar=True)
        wandb.log({'eval/loss': loss})
        self.log('eval/perplexity', perplexity, prog_bar=True)
        wandb.log({'eval/perplexity': perplexity})
        return {'val_loss': loss, 'val_perplexity': perplexity}


# class PetraRQalaLinformer(pl.LightningModule):
#     def __init__(
#             self,
#             d_model,
#             num_tokens,
#             seq_length,
#             depth,
#             k=256,
#             heads=8,
#             dim_head=None,
#             one_kv_head=False,
#             share_kv=True,
#             dropout=0.1,
#             steps=1000,
#     ):
#         super(PetraRQalaLinformer, self).__init__()
#
#         self.d_model = d_model
#         self.num_tokens = num_tokens
#         self.seq_length = seq_length
#         self.depth = depth
#         self.k = k
#         self.heads = heads
#         self.dim_head = dim_head
#         self.one_kv_head = one_kv_head
#         self.share_kv = share_kv
#         self.dropout = dropout
#         self.steps = steps
#
#         self.embeddings = RelativeLogitPositionalEncoding(
#             d_model=d_model,
#             num_embeddings=num_tokens,
#             seq_length=seq_length,
#         )
#
#         self.petrarq_layers = nn.ModuleList()
#         for _ in range(depth):
#             attention = LinformerSelfAttention(
#                 d_model,
#                 seq_len=seq_length,
#                 k=k,
#                 heads=heads,
#                 dim_head=dim_head,
#                 one_kv_head=one_kv_head,
#                 share_kv=share_kv,
#                 dropout=dropout,
#             )
#
#             ff = FeedForward(
#                 d_model,
#                 dropout=dropout,
#                 glu=True,
#             )
#
#             self.petrarq_layers.append(
#                 nn.ModuleList([
#                     PreNorm(d_model, attention),
#                     PreNorm(d_model, ff),
#                 ])
#             )
#
#             self.net = ReversibleSequence(self.petrarq_layers)
#             self.outs = nn.Linear(d_model, num_tokens)
#
#     def forward(self, x):
#         x, position_encodings = self.embeddings(x)
#         x = self.net(x)
#         x = self.outs(x)
#         return x
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adagrad(
#             self.parameters(),
#             lr=1e-6,
#             weight_decay=0.01,
#         )
#
#         lr_scheduler = OneCycleLR(
#             optimizer,
#             max_lr=1e-3,
#             total_steps=self.steps,
#             cycle_momentum=False,
#         )
#
#         return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]
#
#     def training_step(self, batch, batch_idx):
#         x, y, token_pos = batch
#         x = self.forward(x)
#         # print('x shape', x.shape)
#         # print('y shape', y.shape)
#         # print(token_pos)
#         # print('output shape', x[:, -1, :].shape)
#         loss = F.cross_entropy(x[range(x.shape[0]), token_pos, :], y)
#         self.log('train_loss', loss, prog_bar=True)
#         return {'loss': loss}


if __name__ == '__main__':
    # pre = RelativeLogitPositionalEncoding(d_model=10, num_embeddings=7, seq_length=4)

    input_tensor = torch.tensor([
        [1, 2, 3, 4],
        [3, 4, 5, 6]
    ])

    positions = torch.tensor([
        [0, 1, 2, 3],
        [2, 3, 4, 5]
    ])

    # embeds, poss = output_tensor = pre(input_tensor)

    petra = PetraRQ(
        d_model=512,
        num_tokens=7,
        seq_length=4,
        depth=2,
        k=256,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=True,
        dropout=0.1
    )
    outs = petra(input_tensor)

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=1,
    )
    trainer.fit(petra)
