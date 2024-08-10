## HBT+

Implementation of the Hierarchical Bound Tracing Algorithm (HBT+) in C++, using MPI and OpenMP.
Documentation is available on the [wiki](https://github.com/Kambrian/HBT2/wiki).

## About this repository.

The version hosted in this repository was developed from a fork of the [MPI branch](https://github.com/Kambrian/HBTplus) that contained the original HBT+ code.

Several additions have been made to the code in this version, which primarily address:

 - Improved tracking of subhaloes in hydrodynamical and dark matter-only simulations.
 - Better domain decomposition, which was required to run on the large-scale [FLAMINGO](https://flamingo.strw.leidenuniv.nl/) simulations.
 - Compatibility with [SWIFT](https://swift.strw.leidenuniv.nl/) outputs, which can also account for particle splits.

## Dependencies

Required:

 - `C++` compiler with OpenMP support.
 - `HDF5` library.
 - `MPI` library.
 - `cmake`.

Optional:

 - `GSL` if interested in computing inertia tensors.

In our main runs, we have used: `GCC 13.1.0`, `HDF5 1.12.2`, and `OpenMPI 4.1.4`.

## Compilation

First clone this repository:
```bash
git clone https://github.com/SWIFTSIM/HBTplus
cd HBTplus
```

Create a directory where the `HBT` executable will be generated.
```bash
mkdir build
cd build
```

Generate the Makefile using `CMake`. Several options relevant to your particular setup
can be chosen here, e.g. if it is a DMO or a hydrodynamical simulation, or if the gas thermal energy 
is used is included in its binding energy. We therefore recommend using `ccmake` to see all the options.
```bash
ccmake ../
```

Once the appropiate options have been chosen, and the Makefile generated, you can compile HBT+ as follows:
```bash
make -j
```

## Running

Once the executable has been compiled, `HBT` can be run as follows:
```bash
./HBT <PATH_TO_CONFIG> <START_OUTPUT_NUMBER> <END_OUTPUT_NUMBER>
```
`<PATH_TO_CONFIG>` is the path to a configuration text file containing information about the run. See [configs](configs) for 
example configuration files to get started. `<START_OUTPUT_NUMBER>` and `<END_OUTPUT_NUMBER>` are optional arguments that 
are passed when a subset of the simulation outputs are be analysed. For example, when restarting an `HBT` analysis, or when
not all the simulation data is available when starting to run.
