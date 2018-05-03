#!/bin/bash
#SBATCH -p general
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
#SBATCH --mem 128000 # memory pool for all cores
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH -o slurm.cnn2TT.%N.%j.out # STDOUT
#SBATCH -e slurm.cnn2TT.%N.%j.err # STDERR
python CNN2_trainTest.py
