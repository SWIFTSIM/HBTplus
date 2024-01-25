#!/bin/env python

from math import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

import numpy as np
import matplotlib.pyplot as plt

import read_hbt_mpi as rhm
import mass_function as mf


def nr_subplots(nr_plots):
    nr_x = ceil(sqrt(nr_plots))
    nr_y = nr_x
    while nr_x * nr_y < nr_plots:
        nr_y += 1
    return nr_x, nr_y

        
def plot_mass_function(basedirs, names, snap_nr, min_mass, max_mass, nr_bins):

    # Separate out the run directories and names
    basedirs = basedirs.split(",")
    names = names.split(",")

    # Pick mass bins
    bins = np.logspace(np.log10(min_mass), np.log10(max_mass), nr_bins)
    
    # Loop over runs to plot
    for i, (name, basedir) in enumerate(zip(names, basedirs)):
    
        # Read the subhalo catalogue for this run
        subhalo = rhm.read_hbtplus_subhalos(basedir, snap_nr, comm)
        
        # Plot total mass function
        centres, counts =  mf.mass_function(subhalo, bins, comm=comm)
        if comm_rank == 0:
            linestyle = f"C{i}-"
            plt.plot(centres, counts, linestyle, label=name+" (all)")

        # Plot satellite mass function
        centres, counts =  mf.mass_function(subhalo, bins, type="satellite", comm=comm)
        if comm_rank == 0:
            linestyle = f"C{i}:"
            plt.plot(centres, counts, linestyle, label=name+" (sats)")

        # Plot central mass function
        centres, counts =  mf.mass_function(subhalo, bins, type="central", comm=comm)
        if comm_rank == 0:
            linestyle = f"C{i}--"
            plt.plot(centres, counts, linestyle, label=name+" (centrals)")
            
    if comm_rank == 0:
        plt.xlabel("Subhalo mass")
        plt.xscale("log")
        plt.ylabel("Number of subhalos")
        plt.yscale("log")
        plt.title(f"Snapshot {snap_nr}")

        
def plot_mass_functions(basedirs, names, snap_nrs, min_mass, max_mass, nr_bins, title):

    snap_nrs = [int(sn) for sn in snap_nrs.split(",")]
    nx, ny = nr_subplots(len(snap_nrs))

    if comm_rank == 0:
        plt.figure(figsize=(8,8))
    
    for i, snap_nr in enumerate(snap_nrs):
        if comm_rank == 0:
            plt.subplot(ny, nx, i+1)
        plot_mass_function(basedirs, names, snap_nr, min_mass, max_mass, nr_bins)
        if i == 0 and comm_rank == 0:
            plt.legend(loc="lower left", fontsize="small")
        
    if comm_rank == 0:
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

        
if __name__ == "__main__":

    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(comm=comm, description="Plot HBTplus subhalo mass function")
    parser.add_argument("--basedirs", type=str, help="Comma separated output dirs (with ./???/SubSnap_???.*.hdf5 files)")
    parser.add_argument("--names", type=str, help="Comma separated run names")
    parser.add_argument("--snap-nrs", type=str, help="Comma separated snapshot numbers to use")
    parser.add_argument("--min-mass", type=float, help="Minimum mass bin")
    parser.add_argument("--max-mass", type=float, help="Maximum mass bin")
    parser.add_argument("--nr-bins", type=int, help="Number of mass bins")
    parser.add_argument("--title", type=str, help="Title for the figure")
    args = parser.parse_args()

    plot_mass_functions(**vars(args))
    
