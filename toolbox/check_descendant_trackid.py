#!/bin/env python
#
# Try to reproduce DescendantTrackId from merger_tree HBTplus branch
#
# Currently only correct for DMO runs and too slow for large runs.
# Runs on FLAMINGO L1000N0900/DMO_FIDUCIAL in a few minutes.
#

import collections

import h5py
import numpy as np


def read_halos(basedir, snap_nr):

    hbt_filenames=basedir+"/{snap_nr:03d}/SubSnap_{snap_nr:03d}.{file_nr}.hdf5"
    subhalos = []
    particles = []
    file_nr = 0
    nr_files = 1
    while file_nr < nr_files:
        with h5py.File(hbt_filenames.format(file_nr=file_nr, snap_nr=snap_nr), "r") as infile:
            nr_files = int(infile["NumberOfFiles"][...])
            subhalos.append(infile["Subhalos"][...])
            particles.append(infile["SubhaloParticles"][...])
        file_nr += 1
    return np.concatenate(subhalos), np.concatenate(particles)


def check_descendant_trackid(basedir, snap_nr):

    snap1 = snap_nr
    snap2 = snap_nr + 1
    
    nr_tracers=10
    
    # Read subhalos at the earlier time
    sub1, part1 = read_halos(basedir, snap1)

    # Read subhalos at the later time
    sub2, part2 = read_halos(basedir, snap2)

    # Sort all by TrackId
    order = np.argsort(sub1["TrackId"])
    sub1 = sub1[order]
    part1 = part1[order]
    order = np.argsort(sub2["TrackId"])
    sub2 = sub2[order]
    part2 = part2[order]

    # Tracks which exist in both snaps now have the same array index in sub1, sub2
    # but sub2 may have new tracks at the end.
    n1 = len(sub1)
    assert np.all(sub1["TrackId"]==sub2["TrackId"][:n1])
    
    # At the later time, make array of sorted particle IDs with associated TrackId.
    # Skip orphans, because we don't use them in merger tree construction.
    all_partids  = []
    all_trackids = []
    for (s, p) in zip(sub2, part2):
        if s["Nbound"] > 1:
            all_partids.append(p)
            all_trackids.append(np.ones(len(p), dtype=int)*s["TrackId"])
    all_partids = np.concatenate(all_partids)
    all_trackids = np.concatenate(all_trackids)
    order = np.argsort(all_partids)
    all_partids = all_partids[order]
    all_trackids = all_trackids[order]
    assert len(all_partids) == len(all_trackids)

    # Loop over earlier subhalos
    nr_ok = 0
    nr_wrong = 0
    for index, (s1, s2, p1) in enumerate(zip(sub1, sub2[:n1], part1)):

        # Skip orphans
        if s1["Nbound"] <= 1:
            continue
        
        # Initialize dict of possible descendants
        descendants = collections.defaultdict(int)

        # Locate the nr_tracers most bound particles in the sorted array.
        # Some might not be found because they're not in a halo.
        # Some particles might be duplicated if they belong to multiple halos.
        i1 = np.searchsorted(all_partids, p1[:nr_tracers], side="left")
        i2 = np.searchsorted(all_partids, p1[:nr_tracers], side="right")

        # Update counts
        # Loop over tracers in the progenitor
        for i in range(nr_tracers):
            # Loop over later time particles with matching IDs (may have duplicates)
            for j in range(i1[i],i2[i]):
                # Where we have a match, increment count for the later time TrackId
                if all_partids[j] == p1[i]:
                    descendants[all_trackids[j]] += 1

        # Pick descendant:
        # Object with the largest count. Break ties by picking lowest TrackId.
        found_trackids = descendants.keys()
        found_counts = descendants.values()
        descendant_trackid = -1
        descendant_count = 0
        for ft, fc in zip(found_trackids, found_counts):
            if fc > descendant_count:
                descendant_count = fc
                descendant_trackid = ft
            elif (fc == descendant_count) and (ft < descendant_trackid):
                descendant_count = fc
                descendant_trackid = ft
        
        # Check result agrees with DescendantTrackId from HBTplus
        hbt_desc_trackid = s2["DescendantTrackId"]
        if hbt_desc_trackid == descendant_trackid:
            nr_ok += 1
        else:
            nr_wrong += 1

    if nr_ok != np.sum(sub1["Nbound"] > 1):
        raise RuntimeError("Did not find the expected number of matches!")
    if nr_wrong != 0:
        raise RuntimeError("Found mismatches!")

    print(f"Ok, found {nr_ok} matches and {nr_wrong} mismatches")

    
if __name__ == "__main__":

    import sys
    basedir = sys.argv[1]    # Contains 000, 001, 002 etc directories
    snap_nr = int(sys.argv[2])
    
    check_descendant_trackid(basedir, snap_nr)
