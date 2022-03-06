import os
import json
import numpy as np
from itertools import permutations, product
from sklearn.preprocessing import MinMaxScaler

from autolearn.utils import log
from autolearn.feat_selection.nfo.search_space import OpType


class FeatNode:
    def __init__(self, attr, type_, has_order=False, arity=0, order=0, feat_str=None):
        self.attr = attr
        self.has_order = has_order
        self.arity = arity
        self.order = order
        self.children = [None] * arity
        self.feat_str = feat_str
        self.type = type_

    @property
    def child_types(self):
        if self.arity == 0:
            types = []
        elif self.type == OpType.CAT_NUM:
            assert self.arity == 2
            types = [OpType.CAT, OpType.NUM]
        else:
            types = [self.type] * self.arity
        return types

    def join_children(self, children, seq=","):
        ranges = [range(len(child)) for child in children]
        res = []
        for selected in product(*ranges):
            res.append(seq.join([children[child][i] for child, i in enumerate(selected)] + [self.attr]))
        return res

    def post_order(self, full=False):
        orders = []
        children_orders = [child.post_order() for child in self.children]
        if self.has_order or not full:
            orders.extend(self.join_children(children_orders))
        else:
            for order in permutations(range(self.arity)):
                children = [children_orders[i] for i in order]
                orders.extend(self.join_children(children))
        return list(set(orders))

    @property
    def height(self):
        def max_default(seqs, default=0):
            return default if len(seqs) == 0 else max(seqs)

        return 1 + max_default([child.height for child in self.children if child is not None])

    def generate(self, env):
        # feature

        if self.arity == 0:
            return env.get_feat(self.attr)
        else:
            children_feats = [each.generate(env) for each in self.children if each is not None]
            return env.get_op(self.attr)(*children_feats)

    def is_valid(self):
        if self.arity == 0:
            return True
        valid = True
        for i, child_type in enumerate(self.child_types):
            if self.children[i] is None or (not child_type == self.children[i].type):
                valid = False
            else:
                valid = self.children[i].is_valid()
            if not valid:
                break
        return valid

    def __str__(self):
        if self.feat_str is None:
            self.feat_str = self.post_order()[0]
        return self.feat_str


def random_generate_node(attrs, arity, has_order, order, type_dict):
    attr = np.random.choice(attrs, 1)[0]
    type_ = type_dict[attr]
    return FeatNode(attr, type_, has_order=has_order[attr], arity=arity[attr], order=order)


def random_generate_tree(feats, ops, op_arity, op_order, type_dict, max_order=4, max_length=1e3):
    root = random_generate_node(ops, op_arity, op_order, 0, type_dict)
    nodes = [root]
    length = 1
    while len(nodes) > 0 and length < max_length:
        node = nodes.pop()
        if node.order >= max_order:
            continue
        for i, child_type in enumerate(node.child_types):
            tmp_feats = [ele for ele in feats if type_dict[ele] == child_type]
            tmp_ops = [ele for ele in ops if type_dict[ele] == child_type]
            tmp_attrs = tmp_feats + tmp_ops
            if node.order >= max_order - 1:
                selected_attrs = tmp_feats
            # TODO: 0.5 ? something related with order
            elif np.random.randn() > 0.5:
                selected_attrs = tmp_ops
            else:
                selected_attrs = tmp_attrs
            child = random_generate_node(selected_attrs, op_arity, op_order, node.order + 1, type_dict)
            length += 1
            node.children[i] = child
            nodes.append(child)
    return root


def generate_tree_from_str(feats_str, env, delimiter=','):
    trees = []
    type_dict = env.type_dict
    for feat_str in feats_str:
        feat_nodes = feat_str.split(delimiter)[::-1]
        i = 0
        cur = feat_nodes[i]
        op_info = env.op_map.get(cur, (cur, 0, False, None))
        root = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1], order=0, feat_str=feat_str)
        nodes = [root]
        i = 1
        while len(nodes) > 0 and i < len(feat_nodes):
            node = nodes.pop()
            for j in range(node.arity):
                if i + j >= len(feat_nodes):
                    break
                cur = feat_nodes[i + j]
                op_info = env.op_map.get(cur, (cur, 0, False, None))
                child = FeatNode(cur, type_dict[cur], has_order=op_info[2], arity=op_info[1], order=node.order + 1)
                node.children[j] = child
                nodes.append(child)
            i += node.arity
        trees.append(root)
    return trees


def is_valid_feat_str(feat_str, env, delimiter=','):
    if len(feat_str.split(delimiter)) < env.max_length:
        try:
            tree = generate_tree_from_str([feat_str], env, delimiter)[0]
            feat_str = tree.post_order()[0]
            return feat_str, tree.is_valid()
        except Exception as _:
            return feat_str, False
    else:
        return feat_str, False


class FeatureGenerator:
    def __init__(self, env, cur_iter=0, feature_pool=None, population_size=None, ckp_path=None, use_iter=None):
        self.env = env
        self.cur_iter = cur_iter
        self.feature_pool = feature_pool
        if feature_pool is not None and os.path.exists(f"{feature_pool}/{cur_iter}.json"):
            with open(f"{feature_pool}/{cur_iter}.json", 'r') as f:
                self.all = json.load(f)
                log(f"Use features in pool {feature_pool}/{cur_iter}.json")
            with open(f"{feature_pool}/{cur_iter}.meta", 'r') as f:
                self.meta = json.load(f)
                log(f"Use features meta {feature_pool}/{cur_iter}.meta")
        else:
            self.all = self._generate_features(population_size)
            self.meta = {feat: 'explore' for feat in self.all}
        self.population_size = len(self.all)
        self.features, self.scores, self.scaler = None, None, None
        self.init_dataset()
        self.new_feats = []

    def _generate_features(self, population_size):
        env = self.env
        feats, ops, op_arity, op_order, max_order = env.features, env.ops, env.op_arity, env.op_order, env.max_order
        max_length = env.max_length
        random_trees = [
            random_generate_tree(
                feats, ops, op_arity, op_order, env.type_dict, max_order=max_order, max_length=max_length
            ) for _ in range(population_size)
        ]
        random_features = [tree.post_order()[0] for tree in random_trees]
        scores = np.asarray(env.eval_features(random_trees))
        return {feat: score for feat, score in zip(random_features, scores)}

    def init_dataset(self):
        self.features, self.scores = zip(*self.all.items())
        self.features = list(self.features)
        self.scores = np.asarray(self.scores)
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            if "" not in self.all:
                self.all[''] = self.env.eval_set([])
            self.scaler.fit([[self.all[""]], [max(self.scores)]])
        self.scores = self.scaler.transform(self.scores.reshape(-1, 1)).reshape(-1)

    @property
    def dataset(self):
        return self.features, self.scores

    def append_new(self, features):
        cur = set(self.features).union(set(self.new_feats))
        is_changed = np.zeros(len(features), dtype=np.bool)
        valid_features = []
        for i, ele in enumerate(features):
            ele, valid = is_valid_feat_str(ele, self.env)
            if ele not in cur and valid:
                valid_features.append(ele)
                cur.add(ele)
                is_changed[i] = True
        self.new_feats.extend(valid_features)
        self.meta.update({feat: 'exploit' for feat in valid_features})
        return is_changed

    def __len__(self):
        return len(self.features)

    @property
    def new_feat(self):
        return len(self.new_feats)

    def append(self, features):
        features_str = features
        features = generate_tree_from_str(features, self.env)
        scores = self.env.eval_features(features)
        valid_indices = scores != -1
        features_str = [features_str[i] for i, valid in enumerate(valid_indices) if valid]
        scores = scores[valid_indices]
        log(f"scores of new features: {sorted(scores, reverse=True)[:10]}")
        scores = self.scaler.transform(np.asarray(scores).reshape(-1, 1)).reshape(-1)
        self.features.extend(features_str)
        self.scores = np.concatenate([self.scores, scores], axis=-1)

    def get_topk(self, num, duplicated=True):
        indices = np.argsort(self.scores)[::-1]
        if duplicated:
            top_k = []
            i = 0
            cur = set()
            while len(top_k) < num and i < len(self.features):
                index = indices[i]
                feature = self.features[index]
                feature_str = ','.join(set(feature.split(',')))
                if feature_str not in cur:
                    top_k.append(feature)
                    cur.add(feature_str)
                i += 1
        else:
            top_k = [self.features[i] for i in indices[: num]]
        return top_k

    def save(self, incremental=False):
        if incremental:
            indices = np.argsort(self.scores)[::-1]
            indices = indices[:self.population_size]
        else:
            indices = np.arange(len(self.scores))
        scores = self.scaler.inverse_transform(self.scores.reshape(-1, 1)).reshape(-1)
        final = {self.features[i]: scores[i] for i in indices}
        if not os.path.exists(self.feature_pool):
            os.makedirs(self.feature_pool)
        with open(f"{self.feature_pool}/{self.cur_iter + 1}.json", 'w+') as f:
            json.dump(final, f)
            log(f"Save features in pool {self.feature_pool}/{self.cur_iter + 1}.json")
        with open(f"{self.feature_pool}/{self.cur_iter + 1}.meta", 'w+') as f:
            json.dump(self.meta, f)

    def pad_new_feats(self, num):
        exploit = self.new_feat
        explore = num - exploit
        env = self.env
        feats, ops, op_arity, op_order, max_order = env.features, env.ops, env.op_arity, env.op_order, env.max_order
        max_length = env.max_length
        cur = set(self.features).union(set(self.new_feats))
        while self.new_feat < num:
            tree = random_generate_tree(
                feats, ops, op_arity, op_order, env.type_dict, max_order=max_order, max_length=max_length
            )
            feat_str = tree.post_order()[0]
            if feat_str not in cur:
                self.new_feats.append(feat_str)
                self.meta[feat_str] = 'explore'
                cur.add(feat_str)
        log(f"Feature generator:\nexploit {exploit} features\nexplore {explore} features")


def _main():
    from collections import defaultdict
    from search_space import all_op_info
    feats = ["a", "b", "c", "d"]
    op_info = all_op_info
    ops = [op[0] for op in op_info]
    op_arity = defaultdict(lambda: 0)
    op_order = defaultdict(lambda: False)
    type_dict = defaultdict(lambda: 0)
    type_dict.update({op[0]: op[4] for op in op_info})
    type_dict.update({
        "a": OpType.NUM,
        "b": OpType.CAT,
        "c": OpType.NUM,
        "d": OpType.CAT
    })
    for op in op_info:
        op_arity[op[0]] = op[1]
        op_order[op[0]] = op[2]
    tree = [
        random_generate_tree(
            feats, ops, op_arity, op_order, type_dict, max_order=3
        ) for i in range(500)]
    print(tree)


if __name__ == '__main__':
    _main()
