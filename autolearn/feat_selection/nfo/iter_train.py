import argparse
import torch
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split


from autolearn.feat_selection.nfo.feat_tree import generate_tree_from_str, FeatureGenerator
from autolearn.feat_selection.nfo.tokenizer import Tokenizer
from autolearn.feat_selection.nfo.controller import NFOController, Mode, load_nfo
from autolearn.utils import Config, multi_train, timeit, log, get_time_budget, set_time_budget
from autolearn.feat_selection.env import SklearnEnv
from autolearn.feat_selection.dataset import MixDataset

is_cuda = None
device = None


def parse_args():
    parser = argparse.ArgumentParser(description='NFO')
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--cuda', type=str, default="0", help='which gpu to use')
    parser.add_argument(
        '--parallel_eval', type=bool, default=False, help="parallel evaluation when evaluate new features"
    )
    parser.add_argument(
        '--ckp_path', type=str, default=None, help="checkpoint path when training"
    )
    parser.add_argument('--cv', type=int, default=5, help='evaluate cv')
    parser.add_argument('--hyper_config', type=str, default='default')
    parser.add_argument('--eval_model', type=str, default='RF')
    parser.add_argument('--feat_pool', type=str, default=None)
    parser.add_argument('--new_feat', type=int, default=32)
    parser.add_argument('--top_feat', type=int, default=96)
    parser.add_argument('--cur_iter', type=int, default=0)
    parser.add_argument('--total_iter', type=int, default=50)
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--c_scoring', type=str, default="f1_micro")
    parser.add_argument('--scratch', type=bool, default=True)
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def set_cuda(cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    global is_cuda, device
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if is_cuda else torch.device('cpu')


def get_config(path):
    config = Config(path)
    return config


@timeit
def main(args):
    log(f"Current iteration {args.cur_iter} for dataset {args.data}")
    # 1. prepare environments
    set_cuda(args.cuda)
    global is_cuda, device
    set_random_seed(args.seed)
    file_path = os.path.abspath(os.path.dirname(__file__))
    global_config = get_config(f"{file_path}/config/{args.hyper_config}.json")
    config = global_config
    log(f"use device {device}")

    env = SklearnEnv(
        args.data, config.max_order,
        model=args.eval_model,
        cv=args.cv, parallel=args.parallel_eval,
        n_estimators=args.n_estimators, scoring=args.c_scoring
    )
    set_time_budget(env.dataset.time_budget)
    time_budget = get_time_budget()

    # 2. initialize tokenizer & controller
    tokenizer = Tokenizer(env.vocabulary, env.max_length)
    nfo = NFOController(
        tokenizer.vocab_size,
        tokenizer.max_length,
        tokenizer.padding_idx,
        tokenizer.start_idx
    )
    if args.scratch:
        nfo, use_iter = nfo, None
    else:
        nfo, use_iter = load_nfo(nfo, args.ckp_path)
    nfo.to(device)

    # 3. train controller when explore & exploit
    #   - use RandomGenerator generate features and evaluate them, finally get dataset F(feat, score)
    #   - Data Augmentation: each feature tree may have many post order representations
    feature_generator = FeatureGenerator(
        env,
        cur_iter=args.cur_iter,
        feature_pool=args.feat_pool,
        population_size=config.random_set_size,
        ckp_path=args.ckp_path,
        use_iter=use_iter
    )
    features, scores = feature_generator.dataset
    seqs = tokenizer.transform(features)

    sample_num = int(min(10, config.random_set_size))
    args.top_feat = min(args.top_feat, len(feature_generator))

    x_train, x_valid, y_train, y_valid = train_test_split(seqs, scores, test_size=0.2, shuffle=True)
    train_set = MixDataset(x_train, y_train)
    valid_set = MixDataset(x_valid, y_valid)
    # mix_set = MixDataset(seqs, scores, seqs, seqs)
    if args.scratch:
        log(f"features: {features}", level='debug')

    #   - use F to train encoder-predictor-predictor
    if args.scratch:
        nfo.set_mode(Mode.EPD)
        nfo.train()
        optimizer = torch.optim.Adam(
            nfo.parameters(), lr=config.lr,
            weight_decay=config.weight_decay
        )
        multi_train(
            train_set, nfo, optimizer,
            torch.nn.functional.mse_loss, torch.nn.functional.nll_loss,
            device,
            trade_off=config.trade_off,
            trade_off_epoch=config.trade_off_epoch,
            ckp_path=args.ckp_path,
            patience=config.patience,
            valid_set=valid_set,
            epochs=config.num_epoch, batch_size=config.batch_size,
            time_budget=time_budget,
            cur_iter=args.cur_iter
        )

    if args.scratch:
        # - test encoder-predictor
        nfo.set_mode(Mode.EP)
        nfo.eval()
        scores_hat = nfo(torch.tensor(seqs, dtype=torch.long, device=device)).detach().cpu().numpy()
        log(f"original feature scores:\n{scores[: sample_num]}")
        log(f"predict feature scores: {scores_hat.reshape(-1).tolist()[: sample_num]}")

        #   - test encoder-decoder
        nfo.set_mode(Mode.ED)
        nfo.eval()
        _, decoded_seqs = nfo(torch.tensor(seqs, dtype=torch.long, device=device))
        decoded_feats = tokenizer.inverse_transform(decoded_seqs)
        log(f"original features:\n{features[:sample_num]}")
        log(f"decode features to:\n{decoded_feats[:sample_num]}")

    # 4. generate good features by NFO
    nfo.set_mode(Mode.INFER)
    nfo.eval()
    log(f"original features generated randomly:\n{features}", level='debug')
    args.top_feat = min(args.top_feat, len(feature_generator.features))
    original_top_feats = feature_generator.get_topk(args.top_feat)
    seqs = tokenizer.transform(original_top_feats)
    epoch = 0
    for opt_eta in np.arange(config.feat_opt_eta, config.feat_opt_eta * 10, config.feat_opt_eta * config.feat_opt_eta_inc):
        _, new_seqs = nfo(
            torch.tensor(seqs, dtype=torch.long, device=device),
            opt_eta=opt_eta,
            score_threshold=config.score_threshold
        )
        new_feats = tokenizer.inverse_transform(new_seqs)
        is_changed = feature_generator.append_new(new_feats)
        # seqs = seqs[~is_changed]
        if feature_generator.new_feat > args.new_feat:
            break
        epoch += 1
        if epoch > config.num_epoch:
            break
    feature_generator.pad_new_feats(args.new_feat)

    log(f"{feature_generator.new_feats}", level='debug')
    feature_generator.append(feature_generator.new_feats)

    num_top_feats = max(args.new_feat, 32)
    log(f"use {num_top_feats} new features with original {len(env.features)} features")

    feature_generator.save()

    scores = []
    max_score = None
    topk = len(env.features)
    for i in range(topk):
        top_i = i + 1
        features = feature_generator.get_topk(top_i, duplicated=True)
        features = generate_tree_from_str(features, env)
        score = env.eval_set(features)
        scores.append([score])
        if max_score is None or score > max_score[0]:
            max_score = (score, top_i)
    log(f"New max score {max_score}, cur iter: {args.cur_iter}")
    log(f"Original score {env.eval_set([])}")
    return (*max_score, args.cur_iter)


@timeit
def iter_main():
    cur_max = None
    _args = parse_args()
    for i in range(_args.total_iter):
        _args.scratch = _args.cur_iter % 10 == 0
        record = main(_args)
        if cur_max is None or cur_max[0] < record[0]:
            cur_max = record
        _args.cur_iter += 1
    print(cur_max)


if __name__ == '__main__':
    iter_main()

