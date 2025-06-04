#!/bin/bash
#
#SBATCH --partition=cpu_8cores   # Debug partition
#SBATCH --qos=cpu_8cores         # Debug QoS level
#SBATCH --job-name=gen       # Job name
#SBATCH -o slurm.%N.%j.out       # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err       # File containing STDERR output

python generate_imgs.py