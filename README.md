# IFT6289 Winter 2023 Course Project: Improving KL Penalty Optimization in RL Fine-Tuning of Language Models


This projects improves the KL penalty optimization in RL finetuning of language model by 1) using a gradient estimator that computes the KL penlty term in the PPO proxy
objective analytically and 2) using analytic KL divergence in the definition of the KL-augmented reward.

This repository is a fork of [RL4LMs](https://github.com/allenai/RL4LMs). Our main implementaion is located at https://github.com/zhixuan-lin/LM-RL-finetune/tree/main/rl4lms/algorithms/ppo_kl.

## Dependency

Create a conda virtual environment:

```
conda create -n RLLM python=3.9
```

Install dependencies:

```
pip install requirements.txt
```

Activate your environemnt

```
conda activate RLLM
```


## Running


```
python scripts/training/train_text_generation.py \
     --tags debug \
     --base_path_to_store_results ./out \
     --config_path scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_kl_fixed.yml \
     --seed 1 \
     alg.args.analytic_kl_grad=True \
     alg.kl_div.analytic_kl_reward=True \
     train_evaluation.n_iters=150 \
     alg.kl_div.target_kl=0.05 \
```



## Citation


If you find this repository, please use the following citations, including the one for the original RL4LM repo:


```bibtex
@misc{Lin2023ImprovingKL,
  title={Improving KL Penalty Optimization in RL Fine-Tuning of Language Models},
  author={Zhixuan Lin},
  url={https://github.com/zhixuan-lin/LM-RL-finetune},
  year={2023}
}
```

```bibtex
@inproceedings{Ramamurthy2022IsRL,
  title={Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization},
  author={Rajkumar Ramamurthy and Prithviraj Ammanabrolu and Kiant{\'e} Brantley and Jack Hessel and Rafet Sifa and Christian Bauckhage and Hannaneh Hajishirzi and Yejin Choi},
  journal={arXiv preprint arXiv:2210.01241},
  url={https://arxiv.org/abs/2210.01241},
  year={2022}
}
```

