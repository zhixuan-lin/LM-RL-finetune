import os
import tqdm
import numpy as np
from pathlib import Path
import os.path as osp
import sys
import itertools
import subprocess
import time
from typing import List, Dict, Any
num_trials = 1  # 3
dry_run = '--dry-run' in sys.argv
from_pixels = False
if from_pixels:
    fname = 'train_pixels.py'
    #fname = 'train_shared_enc.py'
    runtime = "07:59:00"
    # FOR THIS EXPERIMENT ONLY
    #runtime = "15:59:00"
    use_gpu = True
else:
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # fname = 'train.py'
    # fname = 'train_tandem.py'
    # fname = 'train_align_base.py'
    # fname = 'train_align_load.py'
    # fname = 'train_demo.py'
    fname = 'train_merge_fork.py'
    # fname = 'train_nomerge.py'
    # fname = 'train_show.py
    # fname = 'train_unbalanced.py'
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # TODO
    # runtime = "12:00:00"
    #runtime = "71:59:00"
    # runtime = "16:00:00"
    # runtime = "2:00:00"
    # runtime = "1:30:00"
    runtime = "4:00:00"
    # runtime = "2:59:00"
    # runtime = "8:00:00"
    # runtime = "4:15:00"
    # runtime = "7:30:00"
    # runtime = "7:00:00"
    # runtime = "12:00:00"
    # runtime = "40:00:00"
    # FOR THIS EXPERIMENT ONLY
    use_gpu = True


def add_scratch(suffix, scratch_prefix=osp.expanduser('~/scratch/ift6289')):
# def add_scratch(suffix, scratch_prefix=osp.expanduser('/scratch/zxlin')):
    return os.path.join(scratch_prefix, suffix)

# os.makedirs(add_scratch("slurm_logs"), exist_ok=True)
# os.makedirs(add_scratch("slurm_scripts"), exist_ok=True)


# IMPORTANT: seed arg should be in the end! (if present)
num_seeds = 5
rl_grid = [
     {
         "--tags": ['ppo_kl_fixed'],
         '--config_path': ['scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_kl_fixed.yml'],
         "alg.args.analytic_kl_grad": [True],
         "alg.kl_div.analytic_kl_reward": [True],
         "alg.kl_div.target_kl": [0.05, 0.1, 0.2, 0.5],
         "train_evaluation.n_iters": [150],
         "--seed": [i for i in range(num_seeds)],
     },
     {
         "--tags": ['ppo_kl_fixed'],
         '--config_path': ['scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_kl_fixed.yml'],
         "alg.args.analytic_kl_grad": [False],
         "alg.kl_div.analytic_kl_reward": [False],
         "alg.kl_div.target_kl": [0.05, 0.1, 0.2, 0.5],
         "train_evaluation.n_iters": [150],
         "--seed": [i for i in range(num_seeds)],
     },
     {
         "--tags": ['ppo_kl_fixed'],
         '--config_path': ['scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_kl_fixed.yml'],
         "alg.args.analytic_kl_grad": [True],
         "alg.kl_div.analytic_kl_reward": [False],
         "alg.kl_div.target_kl": [0.05, 0.1, 0.2, 0.5],
         "train_evaluation.n_iters": [150],
         "--seed": [i for i in range(num_seeds)],
     },
     {
         "--tags": ['ppo_kl_fixed'],
         '--config_path': ['scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_kl_fixed.yml'],
         "alg.args.analytic_kl_grad": [False],
         "alg.kl_div.analytic_kl_reward": [True],
         "alg.kl_div.target_kl": [0.05, 0.1, 0.2, 0.5],
         "train_evaluation.n_iters": [150],
         "--seed": [i for i in range(num_seeds)],
     },
     # {
     #     "--tags": ['ppo_fixed'],
     #     '--config_path': ['scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_fixed.yml'],
     #     "alg.args.gamma": [0.99, 1.0],
     #     "alg.kl_div.target_kl": [0.05, 0.5],
     #     "--seed": [i for i in range(num_seeds)],
     # },
     # {
     #     "--tags": ['ppo_fixed'],
     #     '--config_path': ['scripts/training/task_configs/imdb_text_continuation/gpt2_ppo_fixed.yml'],
     #     "train_evaluation.n_iters": [200],
     #     "alg.kl_div.target_kl": [0.05, 0.5],
     #     "--seed": [i for i in range(num_seeds)],
     # },

]


def construct_varying_keys(grids):
    all_keys = set().union(*[g.keys() for g in grids])
    merged = {k: set() for k in all_keys}
    for grid in grids:
        for key in all_keys:
            grid_key_value = grid[key] if key in grid else ["<<NONE>>"]
            merged[key] = merged[key].union(grid_key_value)
    #varying_keys = {key for key in merged if len(merged[key]) > 1}
    varying_keys = {key for key in merged}
    return varying_keys


def construct_jobs(grids: List[Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
    """Construct a list of jobs."""
    jobs = []
    for grid in grids:
        """
        {
            'a': [1, 2],
            'b': [3, 4]
        }
        -->
        {
            [{'a': 1}, {'a': 2}],
            [{'b': 3}, {'b': 4}],
        }
        """
        individual_options = [[{key: value} for value in values]
                              for key, values in grid.items()]
        product_options = list(itertools.product(*individual_options))
        """
        [
            {'a': 1, 'b': 3},
            {'a': 1, 'b': 4},
            {'a': 2, 'b': 3},
            {'a': 2, 'b': 4},
        ]
        """
        jobs += [{k: v for d in option_set for k, v in d.items()}
                 for option_set in product_options]
    return jobs


def construct_flag_string(job):
    """construct the string of arguments to be passed to the script"""
    flagstring = ""
    for flag in job:
        if isinstance(job[flag], bool):
            if flag.startswith('--'):
                raise ValueError('No supported yet')
            else:
                flagstring = flagstring + " " + flag + "=" + str(job[flag])

            # if job[flag]:
                # flagstring = flagstring + " --" + flag
            # else:
                # flagstring = flagstring + " --" + 'no' + flag
                # print("WARNING: Excluding 'False' flag " + flag)
        else:
            if not flag.startswith('--'):
                flagstring = flagstring + " " + flag + "=" + str(job[flag])
            else:
                flagstring = flagstring + " " + flag + " " + str(job[flag])
    return flagstring

def construct_name(job, varying_keys):
    """construct the job's name out of the varying keys in this sweep"""
    varying_keys = [key for key in varying_keys if 'config_path' not in key]
    return '_'.join(['{}_{}'.format(flag.strip('-'), str(job[flag]))
        for flag in job if flag in varying_keys])




def grid_from_job(job):
    return {k: [v] for k, v in job.items()}



def check_job(save_dir, expected_steps):
    entry_templates = [
        'epoch_{expected_steps}_val_split_predictions.json',
        'epoch_{expected_steps}_test_split_predictions.json',
        'model/pytorch_model.bin',
        'model/config.json',
        # # '{seed}_stats.json',
        # # '{seed}_eval_result.json',
        # 'critic_{seed}',
        # 'target_critic_{seed}',
        # 'actor_{seed}',
        # 'temp_{seed}',
        # 'buffer_01000000_{seed}.pickle',
        # '{seed}/buffer_01000000.pickle',
    ]

    okay = True
    for entry_template in entry_templates:
        if 'expected' in entry_template:
            path = Path(save_dir) / entry_template.format(expected_steps=expected_steps)
        else:
            path = Path(save_dir) / entry_template
        if not path.is_file():
            print(path)
            okay = False
            return False
    # with open(Path(save_dir) / f'{seed}.txt', 'r') as f:
        # data = np.loadtxt(f, ndmin=2)
        # if data[-1][0] != expected_steps:
            # print(f'{data[-1][0]} / {expected_steps}')
            # okay = False
    return okay

jobs = construct_jobs(rl_grid)
varying_keys = construct_varying_keys(rl_grid)
job_specs = []
okay_job_specs = []


for job in tqdm.tqdm(jobs):
    # print(grid_from_job(job))
    jobname = construct_name(job, varying_keys)
    flagstring = construct_flag_string(job)

    #flagstring += ' --out_dir ./out/' + jobname
    scratch_jobname = add_scratch('out/' + jobname)
    #flagstring += ' --seed $SEED'
    save_dir = scratch_jobname
    okay = check_job(save_dir, 149) 
    if not okay:
        pass
        # print(grid_from_job(job))
        # job_specs.append(grid_from_job(job))
    else:
        print(jobname)
        okay_job_specs.append(grid_from_job(job))
        # print(grid_from_job(job))
# print(*okay_job_specs, sep=',')
print('Succeeded: {okay}/{total}'.format(okay=len(okay_job_specs), total=len(jobs)))
