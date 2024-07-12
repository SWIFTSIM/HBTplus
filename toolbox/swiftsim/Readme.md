# Particle splitting

Hydrodynamical simulations in SWIFT have the option to split massive gas particles into multiple, less massive ones. This is done to reduce the mass ratio between particles, which would otherwise enhance spurious numerical effects in the N-body solver due to large mass differences. The regions where splits occur are typically in the centres of gas-rich galaxies, since gas particles gain mass by the accretion of metals from nearby star forming regions.

# Splitting particles in HBT+

Given the history based approach of HBT+, the splitting of particles needs to be accounted for. This is particularly important for satellite galaxies, since the particles created during splits need to retain its association to the host subgroup of its progenitor particle. If the split information is not included, the new particles would by default belong to central subgroup, and hence lead to its artificial loss from the satellite.

# How to generate the required files

By default, any HBT+ analysis on a SWIFT hydrodynamical simulation will default to requiring splitting information. This is not always needed, such as when the simulation was run with particle splitting disabled. If this is the case, add `ParticleSplits 0` to the relevant parameter file to omit the need of splitting information.  

If splits are enabled, the required files are generated and saved in `<PATH_TO_HBT_CATALOGUES>/ParticleSplits/particle_splits_XXXX.hdf5` by running:
```bash
python generate_splitting_information.py <PARAMETER_FILE> <SNAPSHOT_INDEX>
```
The `<PARAMETER_FILE>` is the path to the configuration file used to run HBT+. The value of `<SNAPSHOT_INDEX>` corresponds to the snapshot file being analysed, which is not necessarly the same as snapshot number if only a subset of snapshots are analysed (see the parameter `SnapshotIdList`).

Once all of the snapshots to be analysed with HBT have had their split information generated, the user can run HBT as usual. The split information is not needed after creating the catalogues, so it can be safely removed.
