import argparse
import os

from autolearn.utils import log
from autolearn.feat_selection.env import SklearnEnv
from autolearn.feat_selection.nfo.feat_tree import FeatureGenerator, generate_tree_from_str


def parse_args():
    parser = argparse.ArgumentParser(description="NFO dataset test")
    parser.add_argument('--data', type=str, help='dataset name')
    parser.add_argument('--suffix', default='', type=str, help='dataset suffix')
    parser.add_argument('--n_estimators', type=int, default=10)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--max_order', type=int, default=5)
    parser.add_argument('--feat_pool', type=str, default=None)
    parser.add_argument('--cur_iter', type=int, default=None)
    parser.add_argument('--c_scoring', type=str, default="f1_micro")
    parser.add_argument('--topk', type=int, default=None)
    parser.add_argument(
        '--parallel_eval', type=bool, default=False, help="parallel evaluation when evaluate new features"
    )
    args = parser.parse_args()
    if args.feat_pool is None:
        args.feat_pool = f"{os.environ['HOME']}/nfo/iter/{args.data}{args.suffix}"
    if args.cur_iter is None:
        jsons = list(os.listdir(args.feat_pool))
        args.cur_iter = max([int(ele.split(sep='.')[0]) for ele in jsons])
    return args


def main():
    args = parse_args()
    env = SklearnEnv(
        args.data, args.max_order,
        cv=args.cv, parallel=args.parallel_eval,
        n_estimators=args.n_estimators, scoring=args.c_scoring
    )
    feature_generator = FeatureGenerator(
        env,
        cur_iter=args.cur_iter,
        feature_pool=args.feat_pool
    )
    if args.topk is None:
        args.topk = len(env.features)
    max_score = None
    original_score = env.eval_set([])
    scores = []
    for i in range(args.topk):
        top_i = i + 1
        features = feature_generator.get_topk(top_i, duplicated=False)
        features = generate_tree_from_str(features, env)
        score = env.eval_set(features)
        scores.append([score])
        if max_score is None or score > max_score[0]:
            max_score = (score, top_i)
    log(f"{feature_generator.get_topk(args.topk + 1)}")
    log(f"""
            For dataset {args.data} with {len(env.features)} features
            Original score: {original_score}
            Final score: {max_score[0]} with {max_score[1]} new features
            All scores: {scores}
            with new features iter {args.cur_iter}
        """
    )


if __name__ == '__main__':
    main()
