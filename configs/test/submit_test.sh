#!/bin/bash

#SBATCH --ntasks 2
#SBATCH --cpus-per-task=14
#SBATCH -J HBT_TEST
#SBATCH -o %x.%J.out
#SBATCH -e %x.%J.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH -t 00:20:00

module purge
module load gnu_comp/11.1.0 hdf5/1.12.0 openmpi/4.1.1

echo
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo

# Need to do this so that the catalogue parameter file reflects the absolute path.
# Otherwise the HBT reader will not work.
sed -i "s|hbt_output|$PWD\/test_output|g" config_test.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun -np $SLURM_NTASKS ./test_build/HBT config_test.txt

# Revert to the original value of the path
sed -i "s|$PWD\/test_output|hbt_output|g" config_test.txt