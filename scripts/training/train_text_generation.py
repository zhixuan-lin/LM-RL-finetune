import os
from argparse import ArgumentParser

import yaml
from omegaconf import OmegaConf
from typing import List

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
    SupervisedTrainer,
)
import torch
import numpy as np
import random


def main(
    config_path: str,
    project_name: str,
    experiment_name: str,
    base_path_to_store_results: str,
    entity_name: str,
    log_to_wandb: bool,
    cli_args: List[str],
    seed: int
):

    # Random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load the config file
    # with open(config_path, "r") as fp:
        # config = yaml.safe_load(fp)
    config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_dotlist(
        [arg for arg in cli_args if len(arg.strip()) > 0])
    # Sanity check for first level keys
    assert set(cli_config).issubset(set(config)), f'{set(cli_config)} is not a subset of {set(config)}'
    config = OmegaConf.merge(config, cli_config)
    # Convert to dict
    config = OmegaConf.to_object(config)
    config['seed'] = seed

    # load tracker
    tracker = Tracker(
        base_path_to_store_results,
        config,
        project_name,
        experiment_name,
        entity_name,
        log_to_wandb,
    )

    # instantiate the trainer here
    if "supervised" in config["alg"]["id"]:
        trainer = SupervisedTrainer(
            tokenizer_config=config["tokenizer"],
            datapool_config=config["datapool"],
            alg_config=config["alg"],
            train_eval_config=config["train_evaluation"],
            tracker=tracker,
        )
    else:
        trainer = OnPolicyTrainer(
            tokenizer_config=config["tokenizer"],
            datapool_config=config["datapool"],
            reward_config=config["reward_fn"],
            env_config=config["env"],
            on_policy_alg_config=config["alg"],
            train_eval_config=config["train_evaluation"],
            tracker=tracker,
        )
    trainer.train_and_eval()


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune LM to generate controlled text")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--project_name", type=str, help="WANDB project name", default="rl4lm_exps"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="WANDB experiment name",
        default=None,
    )
    parser.add_argument(
        "--entity_name", type=str, help="WANDB entity name", default=None
    )
    parser.add_argument(
        "--tags", type=str, nargs='*', help="WANDB tags", default=None
    )
    parser.add_argument(
        "--base_path_to_store_results",
        type=str,
        help="Base path to store experiment results",
        default='./debug',
    )
    parser.add_argument(
        "--log_to_wandb", action="store_true", default=True, help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )

    args, rest_args = parser.parse_known_args()

    main(
        args.config_path,
        args.project_name,
        args.experiment_name,
        args.base_path_to_store_results,
        args.entity_name,
        args.log_to_wandb,
        rest_args,
        args.seed
    )
