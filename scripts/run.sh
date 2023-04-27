#!/bin/bash

unset CUDA_VISIBLE_DEVICES
module load miniconda
module load cudatoolkit/11.7
export HF_HOME=$HOME/scratch/huggingface
export WANDB_MODE=online
conda activate rl4lm

# python scripts/training/train_text_generation.py \
    # --tags debug \
    # --base_path_to_store_results $HOME/scratch/ift6289/debug/1 \
    # --config_path scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_kl_fixed.yml \
    # --seed 1 \
    # alg.args.analytic_kl_grad=True \
    # alg.kl_div.analytic_kl_reward=True \
    # train_evaluation.n_iters=200 \
    # alg.kl_div.target_kl=0.05 \
    # env.n_envs=1 \
    # alg.args.batch_size=2 \
    # alg.args.gae_lambda=1.0 \
    # alg.args.gamma=1.0 \

python scripts/training/train_text_generation.py --tags debug --config_path scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_kl_fixed.yml alg.args.analytic_kl_grad=False alg.kl_div.analytic_kl_reward=False alg.kl_div.target_kl=0.05 train_evaluation.n_iters=150 --base_path_to_store_results $HOME/scratch/ift6289/debug/1 \
