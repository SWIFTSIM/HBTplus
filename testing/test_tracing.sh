module purge
module load gnu_comp/11.1.0 openmpi/4.1.1 python/3.10.1

# Example path for COLIBRE 
indir=/cosma7/data/dp004/dc-foro1/colibre/hbt_testing/removing_duplicates_no_mergers
snap_files="/cosma7/data/dp004/dc-foro1/colibre/colibre_{snap_nr:04d}.hdf5";

# Example path for FLAMINGO 
indir=/cosma8/data/dp004/dc-foro1/hbt_runs/groups_FLAMINGO_L1000N0900
snap_files="/cosma8/data/dp004/jlvc76/FLAMINGO/FOF/L1000N0900/HYDRO_FIDUCIAL/fof_snapshot/flamingo_{snap_nr:04d}.hdf5";

mpirun -np 32 python3 -m mpi4py ./check_presence_duplicate_particles.py "${indir}" $i;
echo
mpirun -np 32 python3 -m mpi4py ./check_consistency_particles_orphan_subgroups.py "${indir}" 76 76 --snapshot-file="${snap_files}";
echo 
mpirun -np 32 python3 -m mpi4py ./check_tracing_particles_orphan_subgroups.py "${indir}" 76 76 --snapshot-file="${snap_files}";
echo 
mpirun -np 32 python3 -m mpi4py ./check_tracing_particles_resolved_subgroups.py "${indir}" 76 76 --snapshot-file="${snap_files}";
