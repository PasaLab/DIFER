import numpy as np
import pandas as pd
from enum import Enum
from sklearn.preprocessing import MinMaxScaler


class OpType:
    NUM = 1
    CAT = 2
    CAT_NUM = 3


# 一阶数值
def log(ele):
    while np.any(ele == 0):
        ele[ele == 0] += 1e-5
    return np.log(np.abs(ele))


def sqrt_abs(ele):
    return np.sqrt(np.abs(ele))


def min_max(ele):
    scaler = MinMaxScaler()
    return np.squeeze(scaler.fit_transform(np.reshape(ele, (-1, 1))))


def reciprocal(ele):
    while np.any(ele == 0):
        ele[ele == 0] += 1e-5
    return np.reciprocal(ele)



unary_num_op_info = [
    ('log', 1, False, log, OpType.NUM),
    ('sqrt_abs', 1, False, sqrt_abs, OpType.NUM),
    ('min_max', 1, False, min_max, OpType.NUM),
    ('reciprocal', 1, False, reciprocal, OpType.NUM),
]


# 二阶数值特征
def plus(lhs, rhs):
    return lhs + rhs


def minus(lhs, rhs):
    return lhs - rhs


def multiply(lhs, rhs):
    return lhs * rhs


def division(lhs, rhs):
    while np.any(rhs == 0):
        rhs[rhs == 0] += 1e-5
    return np.squeeze(lhs / rhs)


def mod_column(lhs, rhs):
    valid_indices = ~(rhs == 0)
    res = np.zeros_like(lhs, dtype=np.float)
    res[valid_indices] = np.mod(lhs[valid_indices], rhs[valid_indices])
    return res


binary_num_op_info = [
    ('+', 2, False, plus, OpType.NUM),
    ('-', 2, True, minus, OpType.NUM),
    ('*', 2, False, multiply, OpType.NUM),
    ('division', 2, True, division, OpType.NUM),
    ('mod_column', 2, True, mod_column, OpType.NUM)
]


# 二阶cat特征
def count_hash(lhs, rhs):
    hash_fe = lhs + rhs + lhs * rhs
    return pd.DataFrame(hash_fe).fillna(0).groupby([0])[0].transform('count').values


def nunique(lhs, rhs):
    feat = np.concatenate([np.reshape(lhs, (-1, 1)), np.reshape(rhs, (-1, 1))], axis=1)
    return pd.DataFrame(feat).fillna(0).groupby(0)[1].transform('nunique').values


binary_cat_op_info = [
    ('count_hash', 2, False, count_hash, OpType.CAT),
    ('nunique', 2, False, nunique, OpType.CAT),
]


# 二阶num特征
def cat2num_mean(lhs, rhs):
    feat = np.concatenate([np.reshape(lhs, (-1, 1)), np.reshape(rhs, (-1, 1))], axis=1)
    return pd.DataFrame(feat).fillna(0).groupby(0)[1].transform('mean').values


binary_cat_num_op_info = [
    ('cat2num_mean', 2, True, cat2num_mean, OpType.CAT_NUM),
]


num_op_info = unary_num_op_info + binary_num_op_info
cat_op_info = binary_cat_op_info + binary_cat_num_op_info
all_op_info = num_op_info + cat_op_info
