#!/bin/bash -l
#
# Compute which particles have split between two consecutive snapshot indicies.
# This script will generate an HDF5 file that is read by HBT+ and is used to split
# particle.
#
# Specify the path to the configuration file that will be used for the HBT+ run, and
# which snapshot index to do. The latter is specified by the array job index.
#
# After specifying where the configuration file is (via CONFIG_FILE_PATH; see below),
# you can submit it as (for an example that will have 128 HBT+ outputs):
#
# mkdir logs
# sbatch --array=0-128 generate_splitting_information.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 0:10:00

set -e

# This is assuming we run in COSMA
module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source CURRENT_PWD/../openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Path to the HBT configuration file to be analysed
echo "Using configuration present in ${1}" 

# Run the code
mpirun -- python3 -u -m mpi4py \
  CURRENT_PWD/../particle_splitting/generate_splitting_information.py \
  "${1}" \
  ${SLURM_ARRAY_TASK_ID} \
  HBT_FOLDER/splits

echo "Job complete!"
