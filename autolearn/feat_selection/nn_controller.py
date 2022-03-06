import itertools
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from torch import nn


Action = namedtuple("Action", ["op", "op_in"])


class PGController(nn.Module):

    def __init__(self,
                 feature_num,
                 op_num,
                 op_arity,
                 hidden_dim=32,
                 num_lstm_layer=1
                 ):
        super().__init__()
        self.feature_num = feature_num
        self.op_num = op_num
        self.op_arity = op_arity
        self.hidden_dim = hidden_dim
        self.num_lstm_layer = num_lstm_layer
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # 两个embedding需要动态加入到optimizer中
        self.feature_embedding = [torch.rand(1, hidden_dim, requires_grad=True) for _ in range(feature_num)]
        self.op_embedding = torch.rand(hidden_dim, op_num, requires_grad=True)

    def forward(self, inputs, states):
        hx, cx = self.lstm(inputs, states)
        return hx, (hx, cx)

    def sample(self, max_order, batch_size=4):
        entropies = []
        log_probs = []
        actions = []
        inputs, (h, c) = self.init_state(batch_size)
        for i in range(max_order):
            # 1. select feature
            feature_logits, (h, c) = self(inputs, (h, c))
            all_features = torch.cat(self.feature_embedding)
            feature_logits = torch.mm(feat_logits, all_features.transpose(1, 0))
            feature_prob = feature_logits.softmax(dim=-1)
            # 1.1 mixed features
            inputs = torch.mm(feature_prob, all_features)

            # 2. select op
            op_logits, (h, c) = self(inputs, (h, c))
            op_logits = torch.mm(op_logits, self.op_embedding)
            op_probs = op_logits.softmax(dim=-1)
            log_op_probs = F.log_softmax(op_logits, dim=-1)
            entropy = -(log_op_probs * op_probs).sum(1)
            selected_op = torch.multinomial(op_logits, 1).type(torch.long)
            selected_op_logits = log_op_probs.gather(1, selected_op)[:, 0]
            entropies.append(entropy)
            log_probs.append(selected_op_logits)

            inputs = self.op_embedding[selected_op[:, 0]]

            # 3. use ops, features to calculate new feature embedding
            new_features, (h, c) = self(inputs, (h, c))
            self.feature_embedding.extend(new_features.split(split_size=1))

        return actions, torch.cat(entropies, dim=0), torch.cat(log_probs, dim=0)

    def init_state(self, batch_size):
        inputs = torch.rand(batch_size, self.hidden_dim)
        h0 = torch.rand(batch_size, self.hidden_dim)
        c0 = torch.rand(batch_size, self.hidden_dim)
        return inputs, (h0, c0)


def pg_train(optimizer, op_logits, actions, rewards, lambd=0.5, gamma=0.99):
    """
    Use TD($\lambda$) rewards
    Args:
        optimizer: the optimizer of the model
        op_logits: [max_order, num_feature, num_op]
        actions: [max_order, (op, op_in)]
        rewards: ndarray of shape (batch_size, max_order)
        lambd: $\lambda$-return
        gamma: discount factor

    Returns:

    """
    batch_size, order = rewards.shape
    for t in range(order):
        base = rewards[:, t:]
        g_t_lambd = np.zeros_like(rewards[:, t], dtype=np.float)
        for step in range(order - t):
            g_t_n = base[:, 0: step + 1]
            gammas = np.power(gamma, np.arange(0, g_t_n.shape[1]))
            g_t_n = np.sum(g_t_n * gammas, axis=1)
            g_t_n *= np.power(lambd, step)
            g_t_lambd += g_t_n
        rewards[:, t] = (1 - lambd) * g_t_lambd\
                        + np.power(lambd, order - t) * np.sum(base * np.power(gamma, np.arange(0, base.shape[1])), axis=1)
    optimizer.zero_grad()
    for i, j in itertools.product(range(batch_size), range(order)):
        loss = (-torch.log(op_logits[j][i]))[actions[j].op[i]].sum() * rewards[i, j]
        loss.backward()
    optimizer.step()


if __name__ == '__main__':
    controller = PGController(5, 4, [1, 1, 2, 2])
    outputs, (h, c), op_logits, feat_logits, actions = controller()
    outputs2, (h2, c2), op_logits2, feat_logits2, actions2 = controller(outputs, (h, c))
    print(actions)
    optimizer = torch.optim.Adam(list(controller.parameters()) + controller.feature_embedding)
    optimizer.add_param_group({"params": outputs})
    reward = 1.0
    (-torch.log(op_logits[0])[actions.op[0]].sum() * reward).backward()
