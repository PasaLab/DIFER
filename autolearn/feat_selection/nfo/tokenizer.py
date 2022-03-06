import numpy as np
from sklearn.preprocessing import LabelEncoder


class Tokenizer:
    def __init__(self, vocab, max_length):
        self.vocab = vocab
        self._encoder = LabelEncoder()
        self._encoder.fit(['PAD'] + vocab)
        self.max_length = max_length

    def transform(self, sentences, delimiter=","):
        num_sentences = len(sentences)
        padded = np.zeros((num_sentences, self.max_length), dtype=np.int)
        padded[:, :] = self.padding_idx
        # padded[:, 0] = self.start_idx
        for i, ele in enumerate(sentences):
            ele = ele.split(delimiter)
            length = len(ele)
            padded[i, : length] = self._encoder.transform(ele)
        return padded

    def inverse_transform(self, seqs, delimiter=","):
        sentences = []
        for seq in seqs:
            if self.padding_idx in seq:
                padding = np.argwhere(seq == self.padding_idx)
                if (np.diff(padding, 1) != 1).sum() == 0:
                    seq = seq[: padding.min()]
            # if seq[0] == self.start_idx:
            #     seq = seq[1:]
            sentence = self._encoder.inverse_transform(seq.flatten())
            sentences.append(delimiter.join(sentence))
        return sentences

    @property
    def padding_idx(self):
        return int(self._encoder.transform(['PAD'])[0])

    @property
    def start_idx(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.vocab) + 1
