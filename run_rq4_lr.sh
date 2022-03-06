#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./
dataset=$1
cuda=$2
python ./autolearn/feat_selection/nfo/iter_train.py --eval_model=LR --data=$dataset --cuda=$cuda --hyper_config=default --feat_pool=$HOME/nfo/iter/$dataset --ckp_path=$HOME/nfo/ckps/$dataset