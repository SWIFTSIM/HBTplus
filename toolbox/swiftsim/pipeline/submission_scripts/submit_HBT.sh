#!/bin/bash
#SBATCH --ntasks 16 ##128
#SBATCH --cpus-per-task=8
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH -t 00:05:00

module purge
module load gnu_comp/13.1.0 hdf5/1.12.2 openmpi/4.1.4

echo
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo

# Path to HBT executable
HBT_executable=/cosma/apps/durham/dc-foro1/COLIBRE/HBTplus/build/HBT

echo "Using configuration present in ${1}" 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun "${HBT_executable}" $1 $2 $3
