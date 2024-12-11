#!/bin/bash

#SBATCH --ntasks 2
#SBATCH --cpus-per-task=14
#SBATCH -J HBT_TEST
#SBATCH -p cosma7-rp
#SBATCH -A dp004
#SBATCH -t 00:20:00

module purge
module load gnu_comp/13.1.0 hdf5/1.12.2 openmpi/4.1.4
module load rockport-settings

set -e # Abort if a command exits with non-zero status

echo
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo

# Need to do this so that the catalogue parameter file reflects the absolute path.
# Otherwise the HBT reader will not work.
sed -i "/.*SubhaloPath.*/c\SubhaloPath ${OUTPUTDIR}" config.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun $RP_OPENMPI_ARGS -np $SLURM_NTASKS ./build/HBT config.txt

# Revert to the original value of the path
sed -i '/.*SubhaloPath.*/c\SubhaloPath' config.txt

echo
echo "Job complete."
echo
