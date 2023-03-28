#!/bin/bash

module load miniconda
export HF_HOME=$HOME/scratch/huggingface
export WANDB_MODE=online
conda activate rl4lm

python scripts/training/train_text_generation.py \
    --tags debug \
    --base_path_to_store_results $HOME/scratch/ift6289/debug/3 \
    --config_path scripts/training/task_configs/imdb_text_continuation/gpt2_ppo.yml \
