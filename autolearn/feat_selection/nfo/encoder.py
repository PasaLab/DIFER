import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            padding_idx,
            sentence_emb_size=None,
            embedding_size=512,
            lstm_layers=1,
            dropout=0.5,
    ):
        super(Encoder, self).__init__()
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.sentence_emb_size = embedding_size if sentence_emb_size is None else sentence_emb_size

        self.embedding = nn.Embedding(
            vocab_size + 1,
            embedding_size,
            padding_idx=padding_idx,
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            embedding_size, embedding_size, lstm_layers,
            batch_first=True
        )

        def tmp_size(size):
            return int((int((size - 2) / 2) + 1 - 2) / 2) + 1
        # use LeNet as feature extractor
        self.extractor = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5), padding=(2, 2))),
            ('relu1', nn.LeakyReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c2', nn.Conv2d(6, 16, kernel_size=(5, 5), padding=(2, 2))),
            ('relu2', nn.LeakyReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(16, 12, kernel_size=(5, 5), padding=(2, 2))),
            ('relu3', nn.LeakyReLU()),
            ('flatten', Flatten()),
            ('linear', nn.Linear(
                12 * tmp_size(seq_length) * tmp_size(embedding_size),
                self.sentence_emb_size
            )),
            # ('relu3', nn.LeakyReLU())
        ]))

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, sentence_lens=None):
        # x: (batch, seq_len, input_size)
        embedded = self.embedding(x)
        embedded = self.emb_dropout(embedded)
        if sentence_lens is not None:
            embedded = pack_padded_sequence(embedded, sentence_lens, batch_first=True, enforce_sorted=False)
        out, (ht, ct) = self.rnn(embedded)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=self.seq_length)
        out = F.normalize(out, p=2, dim=-1)
        encoder_outputs = out
        encoder_hidden = ct.squeeze(1)
        # (batch, seq_len, input_size)
        # -> (batch, in_channel, seq_len, input_size) to extract features
        # batch mean
        # x_embedding = (out.sum(dim=1).transpose(1, 0) / sentence_lens).transpose(1, 0)
        x_embedding = self.extract_feature(encoder_outputs)
        # NAO中返回: encoder_outputs, encoder_hidden, arch_emb
        #   其中, outputs & hidden为rnn输出
        #   arch_emb为out现在seq_len维度规约然后归一化, 即平均了各个timestep的out
        return x_embedding, encoder_outputs, encoder_hidden

    def extract_feature(self, x):
        x_embedding = x.sum(dim=1)
        x_embedding = F.normalize(x_embedding, p=2, dim=-1)
        return x_embedding
