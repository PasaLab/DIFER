import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from autolearn.utils import log
from autolearn.utils.torch_attention import Attention


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            padding_idx,
            start_idx,
            lstm_layers=1,
            hidden_size=64,
            dropout=0.5
    ):
        super(Decoder, self).__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.rnn = nn.LSTM(
            hidden_size, hidden_size, lstm_layers,
            batch_first=True
        )
        self.embedding = nn.Embedding(
            vocab_size + 1,
            hidden_size,
            padding_idx=padding_idx
        )
        self.emb_dropout = nn.Dropout(dropout)
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward_step(self, x, hidden, attn):
        out, hidden = self.rnn(x, hidden)
        k, v, attn_mask, valid_length = attn
        out, attn = self.attention(out, k)
        # out = F.normalize(out, dim=-1)
        predicted_softmax = F.log_softmax(
            self.out(out), dim=-1
        )
        return predicted_softmax, hidden

    def forward(self, emb, encoder_outputs, encoder_hidden, attn_mask, target_variable=None):
        # embedding of sentences as decode hidden
        self.attention.set_mask(~attn_mask.unsqueeze(1))
        emb = emb.unsqueeze(0)
        batch_size = encoder_outputs.size()[0]
        # encoder_hidden maybe same because of <PAD>
        decoder_hidden = (emb, emb)
        valid_length = attn_mask.sum(dim=-1)
        decoder_outputs = []
        sentences = []
        log(f"""
            emb {emb.size()}\n{emb}
            encoder_output {encoder_outputs.size()}:\n{encoder_outputs}
            encoder_hidden {encoder_hidden.size()}
        """, level="debug")

        # Start Of Sequence
        decoder_input = torch.tensor(
            [self.start_idx] * batch_size, dtype=torch.long, device=emb.device
        ).view(-1)
        for i in range(-1, self.seq_length - 1):
            # encoder output as decoder_input
            if target_variable is not None and i >= 0:
                decoder_input = target_variable[:, i]
            decoder_input = self.embedding(decoder_input).unsqueeze(1)
            decoder_input = self.emb_dropout(decoder_input)
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input,
                decoder_hidden,
                (encoder_outputs, encoder_outputs, attn_mask, valid_length)
            )
            decoder_outputs.append(decoder_output)
            sentence = decoder_output.squeeze(1).argmax(dim=-1).detach().cpu().numpy()
            sentences.append(sentence)
            if target_variable is None:
                decoder_input = torch.tensor(
                    sentence, dtype=torch.long, device=emb.device
                ).view(-1)
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        sentences = np.vstack(sentences).transpose(1, 0)
        log(f"decode output {decoder_outputs.size()}:\n{decoder_outputs}", level="debug")
        if self.training:
            attn_mask[np.arange(0, valid_length.size()[0]), [ele.item() for ele in valid_length]] = True
            decoder_outputs[~attn_mask] *= 0
        return decoder_outputs, sentences


def main():
    from feat_tree import random_generate_tree
    from tokenizer import Tokenizer
    from collections import defaultdict
    from controller import NFOController, Mode
    from autolearn.feat_selection.dataset import SequenceDataset
    from autolearn.utils import torch_train
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    feats = ["a", "b", "c", "d"]
    op_info = [
        ('+', 2, False),
        ('-', 2, True),
        ('*', 2, False),
        ('log', 1, False)
    ]
    ops = [op[0] for op in op_info]
    op_arity = defaultdict(lambda: 0)
    op_order = defaultdict(lambda: False)
    type_dict = defaultdict(lambda: 0)
    max_length = 2 ** 5 - 1
    for op in op_info:
        op_arity[op[0]] = op[1]
        op_order[op[0]] = op[2]
    sentences = [random_generate_tree(feats, ops, op_arity, op_order, type_dict, max_order=4).post_order()[0] for _ in range(20)]
    tokenizer = Tokenizer(
        feats + [op[0] for op in op_info],
        max_length=max_length
    )
    seqs = tokenizer.transform(sentences)
    nfo = NFOController(
        tokenizer.vocab_size,
        tokenizer.max_length,
        tokenizer.padding_idx,
        tokenizer.start_idx
    )
    nfo.set_mode(Mode.ED)
    nfo.train()
    optimizer = torch.optim.Adam(
        nfo.parameters(), lr=0.0005
    )
    # translate_set = SequenceDataset(all_seqs, all_seqs)
    translate_set = SequenceDataset(seqs, seqs)
    torch_train(
        translate_set, nfo, optimizer, torch.nn.functional.nll_loss, device,
        patience=50,
        epochs=200, batch_size=32,
    )
    #   - test encoder-decoder
    nfo.eval()
    _, decoded_seqs = nfo(torch.tensor(seqs, dtype=torch.long, device=device))
    decoded_feats = tokenizer.inverse_transform(decoded_seqs)
    log(f"original features {sentences}")
    log(f"decode features\nto: {decoded_feats}\nfrom: {decoded_seqs}")


if __name__ == '__main__':
    main()

