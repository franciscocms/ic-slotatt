#!/bin/bash
#
#SBATCH --partition=gpu_min8gb    # Debug partition
#SBATCH --qos=gpu_min8gb          # Debug QoS level
#SBATCH --job-name=ICSAiid        # Job name
#SBATCH -o slurm.%N.%j.out       # File containing STDOUT output
#SBATCH -e slurm.%N.%j.err       # File containing STDERR output

python set_prediction.py