import os
import os.path as osp
import sys
import itertools
import subprocess
from typing import List, Dict, Any
dry_run = '--dry-run' in sys.argv
runtime = "12:00:00"
num_trials = 1
# FOR THIS EXPERIMENT ONLY
use_gpu = True

fname='scripts/training/train_text_generation.py'


def add_scratch(suffix, scratch_prefix=osp.expanduser('~/scratch/ift6289')):
    return os.path.join(scratch_prefix, suffix)

os.makedirs(add_scratch("slurm_logs"), exist_ok=True)
os.makedirs(add_scratch("slurm_scripts"), exist_ok=True)


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


jobs = construct_jobs(rl_grid)
varying_keys = construct_varying_keys(rl_grid)
job_specs = []

if dry_run:
    print("NOT starting {} jobs:\n".format(len(jobs)))
else:
    print("Starting {} jobs:\n".format(len(jobs)))

for job in jobs:
    jobname = construct_name(job, varying_keys)
    flagstring = construct_flag_string(job)

    dependency = None

    slurm_log_dir = add_scratch('slurm_logs/' + jobname)
    os.makedirs(os.path.dirname(slurm_log_dir), exist_ok=True)

    #flagstring += ' --out_dir ./out/' + jobname
    scratch_jobname = add_scratch('out/' + jobname)
    flagstring += ' --base_path_to_store_results ' + scratch_jobname

    jobcommand = "python {}{}".format(fname, flagstring)
    # for repeating trials
    jobcommand += " && scancel --name ${SLURM_JOB_NAME}"

    job_specs.append((jobname, jobcommand, dependency))


i = 0
while i < len(job_specs):
    current_jobs = job_specs[i:i+1]

    for job_spec in current_jobs: print(job_spec[1])

    joint_name = ""
    for job_spec in current_jobs:
        if len(joint_name) > 0: joint_name += "__"
        joint_name += job_spec[0]

    # if len(joint_name) > 200:
    #     print('Cropping jobname by 200')
    #     joint_name = joint_name[:200]

    deps = [j[2] for j in current_jobs]
    if any(deps):
        joint_deps = ':'.join([j[2] for j in current_jobs if j[2] is not None])
    else:
        joint_deps = None

    slurm_script_path = add_scratch('slurm_scripts/' + joint_name + '.slurm')
    slurm_script_dir = os.path.dirname(slurm_script_path)
    os.makedirs(slurm_script_dir, exist_ok=True)
    slurm_log_dir = add_scratch("slurm_logs/" + joint_name)
    os.makedirs(slurm_log_dir, exist_ok=True)

    job_start_command = "sbatch " + slurm_script_path

    with open(slurm_script_path, 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        # slurmfile.write("#SBATCH --account=rrg-bengioy-ad\n")
        slurmfile.write("#SBATCH --open-mode=append\n")
        slurmfile.write("#SBATCH --cpus-per-task=8\n")
        slurmfile.write("#SBATCH --job-name" + "=" + joint_name + "\n")
        slurmfile.write("#SBATCH --mem-per-cpu=6GB\n")
        slurmfile.write("#SBATCH --exclude=kepler5,cn-c034\n")
        if use_gpu:
            slurmfile.write("#SBATCH --gres=gpu:40gb:1\n")
            # slurmfile.write("#SBATCH --gres=gpu:16gb:1\n")
#         slurmfile.write("#SBATCH --constraint=16gb|24gb|48gb\n")
#         slurmfile.write("#SBATCH --constraint=turing|volta\n")
#         slurmfile.write("#SBATCH --exclude=blg[4601-4610]\n")
#         slurmfile.write("#SBATCH --exclude=blg[4701-4710]\n")
#         slurmfile.write("#SBATCH --exclude=blg[4801-4810]\n")
#         slurmfile.write("#SBATCH --array=0-{}\n".format(num_seeds-1))
        slurmfile.write("#SBATCH --output=" + slurm_log_dir + "/%A.out\n")  # %A_%a.out\n")
        slurmfile.write("#SBATCH --error=" + slurm_log_dir + "/%A.out\n")  # %A_%a.out\n")
        slurmfile.write("#SBATCH --dependency=singleton\n")
        slurmfile.write("#SBATCH --time=" + runtime + "\n")
        slurmfile.write("\n")

        slurmfile.write("module --quiet load miniconda\n")
        slurmfile.write("module --quiet load cudatoolkit/11.7\n")
        # slurmfile.write("module --quiet load cudnn/8.1\n")
        slurmfile.write("conda activate rl4lm\n")
        slurmfile.write("unset CUDA_VISIBLE_DEVICES\n")
        slurmfile.write("export WANDB_MODE=offline\n")
        slurmfile.write("export HF_HOME=$HOME/scratch/huggingface\n")
        slurmfile.write("\n")
        #slurmfile.write("SEED=$((SLURM_ARRAY_TASK_ID+0))\n")  # +N if we want to bias the seeds

        for job_i, job_spec in enumerate(current_jobs):
            comm = "{command}".format(command=job_spec[1])
            slurmfile.write(comm)

            slurmfile.write("\n")

    if not dry_run:
        for trial in range(num_trials):
            os.system(job_start_command)

    i += 1
