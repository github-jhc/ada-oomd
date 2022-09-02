#!/usr/bin/env bash

source pre_run.sh

export CUDA_VISIBLE_DEVICES=""

omp_thread=2
actors=4
num_cpus=1

./build/deep_cfr/run_ossbcfr --use_regret_net=true --use_policy_net=true --use_tabular=true --num_gpus=0 \
--num_cpus=$num_cpus --actors=$actors --memory_size=40000 --policy_memory_size=100000 --cfr_batch_size=1000 \
--train_batch_size=128 --train_steps=2 --policy_train_steps=16 \
--inference_batch_size=$actors --inference_threads=$num_cpus --inference_cache=100000 \
--omp_threads=$omp_thread --evaluation_window=10 --first_evaluation=10 --exp_evaluation_window=false --game=leduc_poker \
--cfr_rm_scale=0.002 \
--checkpoint_freq=1000000 --sync_period=1 --max_steps=1000000 --graph_def= --suffix=$RANDOM --verbose=false

