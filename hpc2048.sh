#!/bin/bash

#The job should run on the gpu partition
#SBATCH --partition=gpu

#Assign one gpu
#SBATCH --gres=gpu:tesla:1

#The name of the job is test_job
#SBATCH --job-name=zero

#The job requires 1 compute node (no-parallel nodes)
#SBATCH --nodes=1

#The job requires 1 task per node
#SBATCH --ntasks-per-node=1

#The amount of cpu's per task
#SBATCH --cpus-per-task=16

#The amount of required memory
#SBATCH --mem=20000

#The maximum walltime 15 minutes
#SBATCH --time=10:00:00

#Assign working directory
#SBATCH -D /gpfs/hpchome/earl/2048

#SBATCH -o /gpfs/hpchome/earl/2048/slurm_out/slurm-%A.out  # send stdout to outfile
#SBATCH -e /gpfs/hpchome/earl/2048/slurm_out/slurm-err-%A.out  # send stderr to errfile

# Load necessary modules
module load python-3.6.0
module load cuda4
module load cudnn-6.0

python hpc_pipeline/main.py
