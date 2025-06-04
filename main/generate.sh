#!/bin/bash
#
#SBATCH --partition=debug_8gb   # Debug partition
#SBATCH --qos=debug_8gb         # Debug QoS level
#SBATCH --job-name=gen       # Job name
#SBATCH -o slurm.%N.%j.out       # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err       # File containing STDERR output

python generate_imgs.py