#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./
dataset=$1
cuda=$2
python ./autolearn/feat_selection/nfo/iter_train.py --data=$dataset --cuda=$cuda --hyper_config=rq3 --feat_pool=$3/nfo/rq3/iter/$dataset --ckp_path=$3/nfo/rq3/ckps/$dataset