# DIFER

Code for [**"DIFER: Differentiable Automated Feature Engineering"**](https://openreview.net/forum?id=SrrORxarIgc)

> accepted in 1st Conference on Automated Machine Learning



## Requisite

This code is implemented in PyTorch, and we have tested the code under the environment settings in `requirements.txt`.

## What is in this repository

- `data`: $23/25$ medium-sized datasets that can be pushed to git and their meta information.
- `NFS_sklearn_c`: the open-source implementation of **"Neural Feature Search: A Neural Architecture for Automated Feature Engineering"**.
- `autolearn`:the core coes for DIFER in `autolearn/feat_selection/nfo`, continas the feature optimizer in `controller.py`, the feature space in `search_space.py`, the end-to-end training process in `iter_train.py`, the three forms of feature (i.e., the original form, the parse tree and the traversal string) in `feat_tree.py`.

## How to run it

We provide script files for convenience in conducting experiments.
- `run_iter.sh`: after specifying the dataset and cuda, you can run DIFER to automate feature engineering for Random Forest.
- `run_rq3.sh`: the script for RQ3 in the paper.
- `run_rq4_*.sh`: the script of different machine learning algorithms for RQ4 in the paper.


## Reference Code

- NFS: https://github.com/TjuJianyu/NFS
