## HBT+

New implementation of HBT in C++ . This is the hybrid MPI/OpenMP parallelized version. 
Check the [Hydro](https://github.com/Kambrian/HBT2/tree/Hydro) branch for a pure OpenMP 
version.

Documentation is available on the [wiki](https://github.com/Kambrian/HBT2/wiki).

## About

This HBT+ version is compatible with [SWIFT](https://swift.strw.leidenuniv.nl/) 
outputs, and was originally forked from the [following branch](https://github.com/jchelly/HBTplus/tree/swiftsim_pr). 
Since then, a number of additions enhancing the capabilities of HBT, as well as how it 
interfaces with SWIFT-based data outputs have been added.

Summary of additions/changes:

- Ability to specify which particle types to use as tracers of which FoF group hosts a subgroup/track, via the *TracerTypeParticles* parameter.
- Ability to specify two different gravitational softening values, reflecting the commonplace use in cosmological simulations of a comoving and maximum physical softening values: *SofteningHalo* and *MaxPhysicalSofteningHalo*, respectively.
- Automatic reading of gravitational softenings from SWIFT outputs, preventing the accidental use of incorrect values.
- Parameters.log file now groups the values of related parameters together.

**NOTE**: these are currently under development, and so caution is advised.