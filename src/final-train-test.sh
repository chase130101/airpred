#!/bin/bash
#SBATCH -p shared
#SBATCH -N 1 # number of nodes
#SBATCH -n 32 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 0-03:00 # time (D-HH:MM)
#SBATCH -o slurm.TT.%N.%j.out # STDOUT
#SBATCH -e slurm.TT.%N.%j.err # STDERR
python final_train_test.py --model ridge --dataset ridgeImp
