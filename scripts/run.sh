#!/bin/bash

unset CUDA_VISIBLE_DEVICES
module load miniconda
module load cudatoolkit/11.7
export HF_HOME=$HOME/scratch/huggingface
export WANDB_MODE=online
conda activate rl4lm

python scripts/training/train_text_generation.py \
    --tags debug \
    --base_path_to_store_results $HOME/scratch/ift6289/debug/3 \
    --config_path scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_fixed.yml \
    --seed 0 \
    alg.args.gae_lambda=1.0 \
    # alg.args.gamma=1.0 \
