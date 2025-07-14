#!/bin/bash
#
#SBATCH --partition=gpu_min80gb   # Debug partition
#SBATCH --qos=gpu_min80gb         # Debug QoS level
#SBATCH --job-name=icsaCL        # Job name
#SBATCH -o slurm.%N.%j.out       # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err       # File containing STDERR output

python train.py