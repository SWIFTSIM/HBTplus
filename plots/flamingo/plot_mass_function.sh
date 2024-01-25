#!/bin/bash

module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

name1="Original"
basedir1=/cosma8/data/dp004/jch/HBT-test/pos_vel_changes/original/hbt

name2="Modified"
basedir2=/cosma8/data/dp004/jch/HBT-test/pos_vel_changes/new_pos_vel/hbt

mpirun -np 8 python3 -m mpi4py ./plot_mass_function.py \
       --names="${name1},${name2}" \
       --basedirs="${basedir1},${basedir2}" \
       --snap-nrs=77,70,50,30 \
       --min-mass=1.0e2 \
       --max-mass=1.0e5 \
       --nr-bins=30 \
       --title="L1000N0900/DMO_FIDUCIAL"

