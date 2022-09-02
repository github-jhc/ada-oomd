#!/usr/bin/env bash

source pre_run.sh

export CUDA_VISIBLE_DEVICES=""

./build/deep_cfr/run_deep_cfr --use_regret_net=true --use_policy_net=true --num_gpus=0 \
--num_cpus=1 --actors=128 --memory_size=40000000 --policy_memory_size=40000000 \
--cfr_batch_size=1000 --train_batch_size=1000 --train_steps=400 --policy_train_steps=400  \
--omp_threads=10 --evaluation_window=10 --exp_evaluation_window=true --game=leduc_poker \
--inference_batch_size=128 --inference_threads=1 --inference_cache=100000 \
--checkpoint_freq=100 --max_steps=10000000 --suffix=$RANDOM --cfr_mode=ESCFR --graph_def=
