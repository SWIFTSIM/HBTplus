# Load the modules. We assume we have created the relevant virtual enviroment in COSMA
module purge
module load gnu_comp/13.1.0 hdf5/1.12.2 openmpi/4.1.4
source ../../../toolbox/swiftsim/openmpi-5.0.3-hdf5-1.12.3-env/bin/activate

# Example path for COLIBRE test in COSMA7
base_folder=/cosma8/data/dp004/dc-foro1/colibre/low_res_test/
indir="${base_folder}/HBT" # HBT catalogue
snap_files="${base_folder}/colibre_{snap_nr:04d}.hdf5" # SWIFT snapshots

# Tests whether particles of subgroups are confined to their host group (or are not part of any FOF)
mpirun -np 32 python3 -m mpi4py ./check_interhost_subhalos.py "${indir}" 127 127 --snapshot-file="${snap_files}";
echo

# Tests whether there are any duplicated particles in the HBT catalogues.
mpirun -np 32 python3 -m mpi4py ./check_presence_duplicate_particles.py "${indir}" 127;
echo

# Tests whether the ID of the tracer used for orphaned subgroups remains unchanged.
mpirun -np 32 python3 -m mpi4py ./check_consistency_particles_orphan_subgroups.py "${indir}" 127;
echo

# Tests correctness of assigned host groups for orphaned subgroups.
mpirun -np 32 python3 -m mpi4py ./check_tracing_particles_orphan_subgroups.py "${indir}" 127 127 --snapshot-file="${snap_files}";
echo

# Tests correctness of assigned host groups for resolved subgroups.
mpirun -np 32 python3 -m mpi4py ./check_tracing_particles_resolved_subgroups.py "${indir}" 127 127 --snapshot-file="${snap_files}";
