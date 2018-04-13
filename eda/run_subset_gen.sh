#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes 
#SBATCH -n 1 # number of cores 
#SBATCH --mem 30000 # memory pool for all cores 
#SBATCH -t 0-01:00 # time (D-HH:MM) 
#SBATCH -o slurm.%N.%j.out # STDOUT 
#SBATCH -e slurm.%N.%j.err # STDERR 
python3 make_subset_csv.py

