#!/bin/env python
#
# Compute mass function of HBTplus subhalos
#

import numpy as np


def mass_function(subhalo, bins, type=None, comm=None):
    """
    Compute mass function of subhalos in the specified bins. Array of
    subhalos is distributed over communicator comm.

    Returns the same values on all MPI ranks:

    centres: array of bin centres
    counts:  array of counts in bins
    """

    # Determine communicator to use
    from mpi4py import MPI
    if comm is None:
        comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    # Only use resolved subhalos
    resolved = subhalo["Nbound"] > 1

    # Check for centrals/satellites filter
    if type == "central":
        keep = subhalo["Rank"] == 0
    elif type == "satellite":
        keep = subhalo["Rank"] > 0
    elif type is None:
        keep = np.ones(len(subhalo), dtype=bool)
    else:
        raise ValueError("type parameter must be 'central', 'satellite', or None")
    
    # Make a local histogram
    counts, edges = np.histogram(subhalo["Mbound"][resolved & keep], bins=bins)
    centres = np.sqrt(edges[1:]*edges[:-1])

    # Combine results
    counts = comm.allreduce(counts)
    
    return centres, counts

