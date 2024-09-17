#!/bin/bash -l
#
# Post-process HBT catalogues to join them into a single HDF5 file, with the subhaloes
# sorted in ascending TrackId.
#
# Specify the path to where the HBT outputs are saved, and where the joint catalogues will
# be saved. One can also disable the saving of paricles, which reduces storage footprint. If
# they are to be saved, a path to the snapshot should also be provided.
#
# Using as an example an HBT run with 128 catalogue outputs, this script can be run as follows:
#
# mkdir logs
# sbatch --array=0-128 SortCatalogues.sh
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J HBT-SortOutput
#SBATCH -o ./logs/SortOutput.%A.%a.out
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 1:00:00

# We assume we are in COSMA
module purge
module load python/3.12.4 gnu_comp/14.1.0 openmpi/5.0.3 parallel_hdf5/1.12.3
source ./swiftsim/openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Base directory where the original HBT catalogues are stored.
indir=/cosma8/data/dp004/jch/HBT-tests/orphan_tracers/L1000N0900/HYDRO_FIDUCIAL/hbt

# Directory where the joint catalogues will be saved.
outdir=/cosma8/data/dp004/jch/HBT-tests/orphan_tracers/L1000N0900/HYDRO_FIDUCIAL/hbt-sorted

# Snapshot index to do.
snap_nr=${SLURM_ARRAY_TASK_ID}

# Path to snapshot file. Only required if the joint catalogues contain particle information.
snap_files="/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5"

mpirun -- python3 -m mpi4py ./SortCatalogues.py "${indir}" "${snap_nr}" "${outdir}" --with-particles --snapshot-file="${snap_files}"
