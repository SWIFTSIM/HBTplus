#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -J sort_hbt
#SBATCH -o ./logs/%x.%A.%a
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 1:00:00
#

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

indir=/cosma8/data/dp004/jch/HBT-tests/orphan_tracers/L1000N0900/HYDRO_FIDUCIAL/hbt
outdir=/cosma8/data/dp004/jch/HBT-tests/orphan_tracers/L1000N0900/HYDRO_FIDUCIAL/hbt-sorted
snap_nr=${SLURM_ARRAY_TASK_ID}
snap_files="/cosma8/data/dp004/flamingo/Runs/L1000N0900/HYDRO_FIDUCIAL/snapshots/flamingo_{snap_nr:04d}/flamingo_{snap_nr:04d}.{file_nr}.hdf5"

mpirun python3 -m mpi4py ./sort_hbt_output.py "${indir}" "${snap_nr}" "${outdir}" --with-particles --snapshot-file="${snap_files}"
