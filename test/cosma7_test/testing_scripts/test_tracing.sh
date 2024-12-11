module purge
module load gnu_comp/13.1.0 hdf5/1.12.2 openmpi/4.1.4
source ../../../toolbox/swiftsim/openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Example path for COLIBRE
indir=/cosma8/data/dp004/dc-foro1/hbt_runs/groups_COLIBRE_L25N188
snap_files="/cosma8/data/dp004/dc-foro1/colibre/low_res_test/colibre_{snap_nr:04d}.hdf5";

# Example path for FLAMINGO
# indir=/cosma8/data/dp004/dc-foro1/hbt_runs/groups_FLAMINGO_L1000N0900
# snap_files="/cosma8/data/dp004/jlvc76/FLAMINGO/FOF/L1000N0900/HYDRO_FIDUCIAL/fof_snapshot/flamingo_{snap_nr:04d}.hdf5";

mpirun -np 32 python3 -m mpi4py ./check_interhost_subhalos.py "${indir}" 126 126 --snapshot-file="${snap_files}";
echo
mpirun -np 32 python3 -m mpi4py ./check_presence_duplicate_particles.py "${indir}" 126;
echo
mpirun -np 32 python3 -m mpi4py ./check_consistency_particles_orphan_subgroups.py "${indir}" 126;
echo 
mpirun -np 32 python3 -m mpi4py ./check_tracing_particles_orphan_subgroups.py "${indir}" 126 126 --snapshot-file="${snap_files}";
echo 
mpirun -np 32 python3 -m mpi4py ./check_tracing_particles_resolved_subgroups.py "${indir}" 126 126 --snapshot-file="${snap_files}";
