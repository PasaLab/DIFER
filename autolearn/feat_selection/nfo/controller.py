import os
import torch
import numpy as np
import torch.nn as nn
from enum import Enum
from pathlib import Path

from autolearn.feat_selection.nfo.encoder import Encoder
from autolearn.feat_selection.nfo.decoder import Decoder
from autolearn.feat_selection.nfo.predictor import Predictor
from autolearn.utils import log


Mode = Enum('Mode', ('ED', 'EP', 'EPD', 'INFER'))


def load_nfo(controller, ckp_path):
    use_iter = None
    if ckp_path is not None and os.path.exists(ckp_path) and len(os.listdir(ckp_path)) > 0:
        ckp_path = Path(ckp_path)
        ckps = [ckp for ckp in list(os.listdir(ckp_path)) if ckp.endswith('.ckp')]
        ckp_metrics = np.asarray([
            int(ele.split('.ckp')[0].split('_')[0]) for ele in ckps
        ])
        best_ckp = ckps[ckp_metrics.argmax()]
        use_iter = int(best_ckp.split('_')[0])
        controller.load_state_dict(torch.load(ckp_path / best_ckp))
        log(f"Use NFO controller checkpoint in  {ckp_path / best_ckp}")
    return controller, use_iter


class NFOController(nn.Module):
    def __init__(
            self,
            vocab_size,
            seq_length,
            padding_idx,
            start_idx,
            sentence_emb_size=None,
            embedding_size=512,
            lstm_layers=1,
            dropout=0.5
    ):
        super(NFOController, self).__init__()
        self.encoder = Encoder(
            vocab_size,
            seq_length,
            padding_idx,
            sentence_emb_size=sentence_emb_size,
            embedding_size=embedding_size,
            lstm_layers=lstm_layers,
            dropout=dropout
        )
        self.predictor = Predictor(
            embedding_size
        )
        self.decoder = Decoder(
            vocab_size,
            seq_length,
            padding_idx,
            start_idx,
            lstm_layers=lstm_layers,
            hidden_size=embedding_size
        )
        self.padding_idx = padding_idx
        self.mode = Mode.EPD

        self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def train(self, mode=True):
        super(NFOController, self).train(mode)
        if self.mode == Mode.EP:
            self.decoder.eval()
        elif self.mode == Mode.ED:
            self.encoder.eval()

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, x, opt_eta=1e-3, score_threshold=None):
        attn_mask = x != self.padding_idx
        valid_length = attn_mask.sum(dim=-1) + 1
        # x: (batch_size, seq_len, input_size)
        if self.mode == Mode.ED:
            embedding, out, hidden = self.encode(x, valid_length)
            # avoid RuntimeError: cudnn RNN backward can only be called in training mode
            embedding, out, hidden = embedding.detach(), out.detach(), hidden.detach()
            x, symbols = self.decode(embedding, out, hidden, attn_mask, target_var=x)
            if self.training:
                return x
            else:
                return x, symbols
        elif self.mode == Mode.EP:
            embedding, out, hidden = self.encode(x, valid_length)
            scores = self.predict(embedding)
            return scores
        elif self.mode == Mode.EPD:
            embedding, out, hidden = self.encode(x, valid_length)
            scores = self.predict(embedding)
            x, symbols = self.decode(embedding, out, hidden, attn_mask, target_var=x)
            if self.training:
                return scores, x
            else:
                return scores, symbols
        else:
            embedding, out, hidden = self.infer(x, opt_eta, valid_length, score_threshold)
            attn_mask = x != self.padding_idx
            x, symbols = self.decode(embedding, out, hidden, attn_mask)
            return x, symbols

    def encode(self, x, valid_length):
        embedding, out, hidden_state = self.encoder(x, valid_length)
        return embedding, out, hidden_state

    def predict(self, x):
        x = self.predictor(x)
        return x

    def decode(self, embedding, out, hidden, attn_mask, target_var=None):
        x, symbols = self.decoder(embedding, out, hidden, attn_mask, target_variable=target_var)
        return x, symbols

    def infer(self, x, eta, valid_length, score_threshold):
        # 1. forward, evaluate score of current x
        embedding, encoder_outputs, encoder_hidden = self.encode(x, valid_length)
        score = self.predict(embedding)

        # 2. opt score to generate new better embedding
        cur_score = score.detach().cpu().numpy()
        final_score = cur_score * score_threshold
        new_embeddings = embedding.detach()
        length = len(cur_score)
        epoch = 0
        while np.sum(cur_score > final_score) < length * 0.5 and epoch < 50:
            opt_outputs = torch.autograd.grad(score, encoder_outputs, torch.ones_like(score))[0]
            encoder_outputs.data = encoder_outputs.data + eta * opt_outputs
            new_embeddings = self.encoder.extract_feature(encoder_outputs)
            score = self.predict(new_embeddings)
            cur_score = score.detach().cpu().numpy()
            epoch += 1

        return new_embeddings, encoder_outputs, encoder_hidden


def main():
    from feat_tree import random_generate_tree
    from tokenizer import Tokenizer
    from collections import defaultdict
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
    sentences = [random_generate_tree(feats, ops, op_arity, op_order, type_dict, max_order=4).post_order()[0] for _ in range(10)]
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
    nfo.set_mode(Mode.INFER)
    # nfo.eval()
    log_softmax, new_feat = nfo(torch.tensor(seqs, dtype=torch.long), score_threshold=1.3)
    log(new_feat)
    log(tokenizer.inverse_transform(new_feat))


if __name__ == '__main__':
    main()
