#!/bin/bash

#SBATCH --job-name=HBT
#SBATCH --ntasks <NUMBER_MPI_RANKS>
#SBATCH --cpus-per-task=<NUMBER_OMP_TASKS_PER_MPI_RANK>
#SBATCH -p <PARTITION>
#SBATCH -A <ACCOUNT>
#SBATCH -t <HOURS>:<MINUTES>:<SECONDS>
#SBATCH --output=./logs/%x.%j.out
#SBATCH --error=./logs/%x.%j.err

# Tell HBT how many omp threads we want per MPI task.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load the required modules. Here we assume a cosma-like system.
module purge
module load gnu_comp/13.1.0 hdf5/1.12.2 openmpi/4.1.4

# Folder to store the logs
mkdir -p logs

echo
echo "Running on $SLURM_NNODES hosts: $SLURM_NODELIST"
echo -n "Using on $SLURM_NPROCS MPI ranks, each using $SLURM_CPUS_PER_TASK OMP threads."
echo "Current working directory is `pwd`"
echo

# Path to the HBT executable and configuration to use.
HBT_executable="<PATH_TO_HBT_EXECUTABLE>"
HBT_config="<PATH_TO_HBT_CONFIGURATION>"

# Whether to restart or analyse a subset of the snapshots specified in the configuration file.
START_SNAP="<FIRST_SNAP>"
END_SNAP="<LAST_SNAP>"

# Run!
mpirun $HBT_executable $HBT_config $START_SNAP $END_SNAP
