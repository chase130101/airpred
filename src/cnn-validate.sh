#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH -o slurm.cnnVal.%N.%j.out # STDOUT
#SBATCH -e slurm.cnnVal.%N.%j.err # STDERR
python CNN_validate.py
