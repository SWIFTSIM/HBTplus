#!/bin/bash

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

nr_ranks=8

name1="Original"
basedir1=/cosma8/data/dp004/jch/HBT-test/pos_vel_changes/original/hbt
fofname1="/cosma8/data/dp004/flamingo/Runs/L1000N0900/DMO_FIDUCIAL/fof/fof_output_{snap_nr:04d}/fof_output_{snap_nr:04d}.{file_nr}.hdf5"

name2="Modified"
basedir2=/cosma8/data/dp004/jch/HBT-test/pos_vel_changes/new_pos_vel/hbt
fofname2="/cosma8/data/dp004/flamingo/Runs/L1000N0900/DMO_FIDUCIAL/fof/fof_output_{snap_nr:04d}/fof_output_{snap_nr:04d}.{file_nr}.hdf5"

#
# Plot mass function of all subhalos
#
mpirun -np ${nr_ranks} python3 -m mpi4py ./plot_mass_function.py \
       --names="${name1},${name2}" \
       --basedirs="${basedir1},${basedir2}" \
       --snap-nrs=77,60,40,20 \
       --min-mass=3.0e11 \
       --max-mass=3.0e15 \
       --nr-bins=30 \
       --title="L1000N0900/DMO_FIDUCIAL, all FoF groups" \
       --output-file="mass_function_all.png"

#
# Plot mass function of subhalos in FoF of some mass
#
mpirun -np ${nr_ranks} python3 -m mpi4py ./plot_mass_function.py \
       --names="${name1},${name2}" \
       --basedirs="${basedir1},${basedir2}" \
       --fof-names="${fofname1},${fofname2}" \
       --snap-nrs=77,60,40,20 \
       --min-mass=3.0e11 \
       --max-mass=3.0e15 \
       --min-fof-mass=1.0e13 \
       --max-fof-mass=1.0e14 \
       --nr-bins=30 \
       --title="L1000N0900/DMO_FIDUCIAL, FoF mass 1e13 to 1e14Msolar/h" \
       --output-file="mass_function_fof_1e13_1e14.png"

