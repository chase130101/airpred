#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 128000 # memory pool for all cores
#SBATCH -t 0-24:00 # time (D-HH:MM)
#SBATCH -o slurm.cnn110001Val.%N.%j.out # STDOUT
#SBATCH -e slurm.cnn110001Val.%N.%j.err # STDERR
python CNN1_validate_100_01.py
