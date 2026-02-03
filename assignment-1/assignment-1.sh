#!/bin/bash

#======================================================
#
# Job script for running LAMMPS on multiple nodes
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the teaching partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required
#SBATCH --ntasks=16 
#
# Distribute processes in round-robin fashion for load balancing
#SBATCH --distribution=block:block
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=01:00:00
#
# Job name
#SBATCH --job-name=assignment-1
#
# Output file
#SBATCH --output=slurm-%j.out
#======================================================

module load mpi

perf stat -e cycles,instructions,cache-misses mpirun -np 16 ./assignment-1.py --visualize
