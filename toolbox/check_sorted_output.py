#!/bin/env python
#
# Sanity check on sorted HBT output
#

import h5py
import numpy as np

def check_output(filename):

    with h5py.File(filename, "r") as infile:
        subhalo_nbound = infile["Subhalos"]["Nbound"][...]
        subhalo_mbound = infile["Subhalos"]["Mbound"][...]
        subhalo_nboundtype = infile["Subhalos"]["NboundType"][...]        
        subhalo_offset = infile["Subhalos"]["ParticleOffset"][...]
        particle_type = infile["Particles"]["Type"][...]
        particle_mass = infile["Particles"]["Masses"][...]

    for i in range(len(subhalo_nbound)):

        # Check NboundType is consistent with individual particle types
        ptypes = particle_type[subhalo_offset[i]:subhalo_offset[i]+subhalo_nbound[i]]
        nr_part = np.bincount(ptypes, minlength=6)[:6]
        assert np.all(nr_part == subhalo_nboundtype[i,...])

        # Check Mbound is consistent with sum of particle masses
        if subhalo_nbound[i] > 0:
            pmass = particle_mass[subhalo_offset[i]:subhalo_offset[i]+subhalo_nbound[i]]
            total_mass = np.sum(pmass, dtype=np.float64)
            rel_diff = np.abs(total_mass/subhalo_mbound[i] - 1.0)
            #print(rel_diff)
            assert rel_diff < 1.0e-6
            

if __name__ == "__main__":

    import sys
    check_output(sys.argv[1])
