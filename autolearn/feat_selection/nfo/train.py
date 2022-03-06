import argparse
import itertools
import torch
import random
import os
import numpy as np


from autolearn.feat_selection.nfo.feat_tree import random_generate_tree, generate_tree_from_str
from autolearn.feat_selection.nfo.tokenizer import Tokenizer
from autolearn.feat_selection.nfo.controller import NFOController, Mode
from autolearn.utils import Config, torch_train, timeit, log, get_time_budget, set_time_budget
from autolearn.feat_selection.env import Environment
from autolearn.feat_selection.dataset import PlainDataset, SequenceDataset

is_cuda = None
device = None


def parse_args():
    parser = argparse.ArgumentParser(description='NFO')
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=str, default="0", help='which gpu to use')
    parser.add_argument(
        '--parallel_eval', type=bool, default=True, help="parallel evaluation when evaluate new featres"
    )
    parser.add_argument(
        '--ckp_path', type=str, default=None, help="checkpoint path when training"
    )
    parser.add_argument('--cv', type=int, default=5, help='evaluate cv')
    parser.add_argument('--hyper_config', type=str, default='default')
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def set_cuda(cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    global is_cuda, device
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')


def get_config(path):
    config = Config(path)
    return config


@timeit
def main():
    # 1. prepare environments
    args = parse_args()
    set_cuda(args.cuda)
    global is_cuda, device
    set_random_seed(args.seed)
    file_path = os.path.abspath(os.path.dirname(__file__))
    global_config = get_config(f"{file_path}/config/{args.hyper_config}.json")
    config = global_config
    env = Environment(args.data, config.max_order, cv=args.cv, parallel=args.parallel_eval)
    feats, ops, op_arity, op_order = env.features, env.ops, env.op_arity, env.op_order
    set_time_budget(env.dataset.time_budget)
    time_budget = get_time_budget()

    # 2. initialize tokenizer & controller
    tokenizer = Tokenizer(env.vocabulary, env.max_length)
    nfo = NFOController(
        tokenizer.vocab_size,
        tokenizer.max_length,
        tokenizer.padding_idx,
        tokenizer.start_idx,
        opt_eta=config.feat_opt_eta
    ).to(device)

    # 3. train controller when explore & exploit

    #   - use RandomGenerator generate features and evaluate them, finally get dataset F(feat, score)
    #   - Data Augmentation: each feature tree may have many post order representations
    random_trees = [
        random_generate_tree(
            feats, ops, op_arity, op_order, max_order=config.max_order
        ) for _ in range(config.random_set_size)
    ]
    random_features = [
        tree.post_order() for tree in random_trees
    ]
    unique_features = [ele[0] for ele in random_features]
    seqs = tokenizer.transform(unique_features)
    scores = env.eval_features(random_trees)

    features_map = {order: orders[0] for orders in random_features for order in orders}
    scores_map = {unique_features[i]: scores[i] for i in range(len(scores))}
    all_features = list(itertools.chain(*random_features))
    all_seqs = tokenizer.transform(all_features)
    all_scores = [scores_map[features_map[feat]] for feat in all_features]
    random_set = PlainDataset(all_seqs, all_scores)

    log(f"features: {unique_features}")
    log(f"seqs: {seqs}")
    #   - use F to train encoder-predictor
    nfo.set_mode(Mode.EP)
    nfo.train()
    config = global_config.child_config('ep')
    optimizer = torch.optim.Adam(
        nfo.parameters(), lr=config.lr,
        weight_decay=config.weight_decay
    )
    torch_train(
        random_set, nfo, optimizer, torch.nn.functional.mse_loss, device,
        ckp_path=f"{args.ckp_path}/ep" if args.ckp_path is not None else None,
        patience=config.patience,
        epochs=config.num_epoch, batch_size=config.batch_size,
        time_budget=time_budget
    )
    nfo.eval()
    scores_hat = nfo(torch.tensor(seqs, dtype=torch.long, device=device)).detach().cpu().numpy()
    log(f"original feature scores: {scores}")
    log(f"predict feature scores: {scores_hat.reshape(-1).tolist()}")

    #   - train encoder-decoder
    nfo.set_mode(Mode.ED)
    nfo.train()
    config = global_config.child_config('ed')
    optimizer = torch.optim.Adam(
        nfo.parameters(), lr=config.lr,
        weight_decay=config.weight_decay
    )
    # translate_set = SequenceDataset(all_seqs, all_seqs)
    translate_set = SequenceDataset(seqs, seqs)
    torch_train(
        translate_set, nfo, optimizer, torch.nn.functional.nll_loss, device,
        ckp_path=f"{args.ckp_path}/ed" if args.ckp_path is not None else None,
        patience=config.patience,
        epochs=config.num_epoch, batch_size=config.batch_size,
        time_budget=time_budget
    )
    #   - test encoder-decoder
    nfo.eval()
    _, decoded_seqs = nfo(torch.tensor(seqs, dtype=torch.long, device=device))
    decoded_feats = tokenizer.inverse_transform(decoded_seqs)
    log(f"original features {unique_features}")
    log(f"decode features\nto: {decoded_feats}\nfrom: {decoded_seqs}")

    # 4. generate good features by NFO
    nfo.set_mode(Mode.EPD)
    nfo.eval()
    log(f"original features generated randomly:\n{unique_features}")
    _, new_seqs = nfo(torch.tensor(seqs, dtype=torch.long, device=device))
    new_feats = tokenizer.inverse_transform(new_seqs)
    log(f"all features searched by NFO:\n{new_feats}\n{new_seqs}")
    new_feats_str = new_feats
    new_feats = generate_tree_from_str(new_feats, env)
    final_score = env.eval_set(new_feats)
    log(f"""
        Final score : {final_score}
        with new features {new_feats_str}
    """)


if __name__ == '__main__':
    # NAO的好处在于, 不用管特征生成的问题, 只要decoder生成合法序列
    # 序列是特征生成树的前序表示, 利用数据增强加强序列
    #   1. 类似EA进化算法, 随机生成一批合法feature seq
    #       1.1 树结构, 前序seq, 需要尽可能利用所有特征
    #   2. 通过encoder进行embedding
    #   3. 利用predictor预测, 数据增强
    #   4. 训练decoder
    #   5. semi-nfo 通过半监督方式进行训练
    main()
