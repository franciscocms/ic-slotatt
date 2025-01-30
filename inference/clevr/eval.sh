#!/bin/bash
#
#SBATCH --partition=gpu_min11gb   # Debug partition
#SBATCH --qos=gpu_min11gb         # Debug QoS level
#SBATCH --job-name=eval_3        # Job name
#SBATCH -o slurm.%N.%j.out       # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err       # File containing STDERR output

python eval.py