import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .tools import log


def sequence_mask(X, X_len, value=-1e6):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float, device=X.device)[None, :] >= X_len[:, None]
    # print(mask)
    X[mask] = value
    return X


def masked_norm_softmax(X, valid_length=None):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    softmax = nn.Softmax(dim=-1)
    if valid_length is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_length.dim() == 1:
            try:
                valid_length = torch.tensor(
                    valid_length.numpy().repeat(shape[1], axis=0),
                    dtype=torch.float,
                    device=valid_length.device
                )  # [2,2,3,3]
            except:
                valid_length = torch.tensor(
                    valid_length.cpu().numpy().repeat(shape[1], axis=0),
                    dtype=torch.float,
                    device=valid_length.device
                )  # [2,2,3,3]
        else:
            valid_length = valid_length.reshape((-1,))
        # fill masked elements with a large negative, whose exp is 0
        X = F.normalize(X - X.min(dim=-1)[0].unsqueeze(2), p=2, dim=-1)
        X = sequence_mask(X.reshape((-1, shape[-1])), valid_length)
        return softmax(X).reshape(shape)


class DotProductAttention(nn.Module):
    def __init__(self, dropout, sharp=1.0, **kwargs):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.sharp = sharp

    # query: (batch_size, #queries, d)
    # key: (batch_size, #kv_pairs, d)
    # value: (batch_size, #kv_pairs, dim_v)
    # valid_length: either (batch_size, ) or (batch_size, xx)
    def forward(self, query, key, value, valid_length=None):
        # d = query.shape[-1]
        # set transpose_b=True to swap the last two dimensions of key
        scores = torch.bmm(query, key.transpose(1, 2)) / self.sharp
        attention_weights = masked_norm_softmax(scores, valid_length)
        # TODO: attention score 接近于平均
        # TODO: softmax 太平滑了
        #   每一轮都没啥变化啊
        log(f"attention_weight\n{attention_weights}", level="info")
        return self.dropout(torch.bmm(attention_weights, value))


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, input, source_hids):
        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn = attn.masked_fill(self.mask, -np.inf)
            # attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn, dim=-1)

        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        # TODO: 加速
        log(f"attention_weight\n{attn}", level="debug")
        mix = torch.bmm(attn, source_hids)

        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=-1)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(
            self.output_proj(
                combined
            )
        )
        return output, attn


class DotAttention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_size, dec_hidden_size)

    def forward(self, hidden, encoder_output, mask):
        # hidden: (layers, batch_size, hidden_size)
        # encoder_output: (batch_size, seq_length, directions*enc_hidden_size)
        # mask: (batch_size, seq_length)
        batch_size, seq_length = encoder_output.shape[0], encoder_output.shape[1]

        # repeat decoder hidden state seq_length times
        # hidden: (batch_size, seq_length, hidden_size)
        hidden = hidden[-1].unsqueeze(1).repeat(1, seq_length, 1)

        # energy: (batch_size, seq_length, dec_hidden_size)
        energy = self.attn(encoder_output)
        # attn_scores: (batch_size, seq_length)
        attn_scores = torch.sum(energy * hidden, dim=-1)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        attn_scores = F.softmax(attn_scores, dim=-1)
        return torch.bmm(attn_scores.unsqueeze(dim=1), encoder_output), attn_scores
