import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(
            self,
            embedding_size,
            mlp_dropout=0.2,
            mlp_layers=5,
            mlp_hidden_size=None
    ):
        super(Predictor, self).__init__()
        self.embedding_size = embedding_size
        self.mlp = nn.Sequential()
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = embedding_size * 2 if mlp_hidden_size is None else mlp_hidden_size
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module(
                    'layer_{}'.format(i),
                    nn.Sequential(
                        nn.Linear(self.embedding_size, self.mlp_hidden_size),
                        nn.LeakyReLU(inplace=False),
                        nn.Dropout(p=mlp_dropout))
                )
            else:
                self.mlp.add_module(
                    'layer_{}'.format(i),
                    nn.Sequential(
                        nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                        nn.LeakyReLU(inplace=False),
                        nn.Dropout(p=mlp_dropout))
                )
        self.regressor = nn.Linear(self.mlp_hidden_size, 1)

    def forward(self, x):
        x = self.mlp(x)
        out = self.regressor(x)
        predict_value = torch.tanh(out)
        return predict_value
