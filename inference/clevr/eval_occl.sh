#!/bin/bash
#
#SBATCH --partition=gpu_min80gb   # Debug partition
#SBATCH --qos=gpu_min80gb         # Debug QoS level
#SBATCH --job-name=eval_occl         # Job name
#SBATCH -o slurm.%N.%j.out       # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err       # File containing STDERR output

python eval_occlusion.py