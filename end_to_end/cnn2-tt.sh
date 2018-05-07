#!/bin/bash
#SBATCH -p general
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 100000 # memory pool for all cores
#SBATCH -t 0-24:00 # time (D-HH:MM)
#SBATCH -o slurm.cnn250TT.%N.%j.out # STDOUT
#SBATCH -e slurm.cnn250TT.%N.%j.err # STDERR
python CNN2_trainTest.py
