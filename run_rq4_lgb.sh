#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./
cuda=$1
log_folder="./lgb_logs"

datasets=("586" "589" "607" "616" "618" "620" "637" "airfoil" "amazon" "bikeshare" "credit_a" "credit_dafault" "fertility" "german" "hepatitis" "housing" "ionosphere" "lymphography" "megawatt1" "messidor_features" "pima" "spambase" "spectf" "winequality-red" "winequality-white")
for dataset in ${datasets[@]}
do
  cur_time="`date +%Y-%m-%d-%H-%M-%S`"
  log_file="$log_folder/$dataset-$cur_time.log"
  log_command="tee -i $log_file"
  python_command="python ./autolearn/feat_selection/nfo/iter_train.py --eval_model=LGB --data=$dataset --cuda=$cuda --hyper_config=default --feat_pool=$HOME/nfo/rq4_lgb/iter/$dataset --ckp_path=$HOME/nfo/rq4_lgb/ckps/$dataset"
  echo "Current time: $cur_time"
  echo "Run command: $python_command"
  echo "Log info into file: $log_file"
  eval "$python_command | $log_command"
done
