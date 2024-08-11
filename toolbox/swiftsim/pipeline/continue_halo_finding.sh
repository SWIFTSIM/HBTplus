#!/bin/bash

set -e

if [ "$#" -ne 1 ]
then
  echo "Usage: ./continue_halo_finding.sh <SIMULATION_NAME>"
  exit 1
fi

# Base folder of the simulation
BASE_FOLDER=${1}

# There should be an HBT folder already there. If not, a path may be wrong.
if [ ! -d $BASE_FOLDER/HBTplus ]; then
    echo "No HBTplus folder found in provided path. Check whether path is correct, and if so, create an HBTplus folder first."
fi

HBT_FOLDER=$BASE_FOLDER/HBTplus

# Check if the configuration file exists in the destination
if [ ! -f $HBT_FOLDER/config.txt ]; then
  echo "No configuration file exists at output folder. This script is only used to continue HBT+ runs. Exiting."
  exit 1
fi

# Where logs for particle splits and HBT will be saved (Should already exist)
mkdir -p $HBT_FOLDER/logs/particle_splits

# We check how many SWIFT particle outputs have been done
MIN_SNAPSHOT=0
MAX_SNAPSHOT=$(find $BASE_FOLDER/$SNAPSHOT_SUBDIR/ -maxdepth 1 -name "colibre_????" | wc -l)

# We check how many HBT catalogues have been done
MAX_HBT_OUTPUT=$(find $BASE_FOLDER/HBTplus -maxdepth 2 -name "SubSnap_???.0.hdf5" | wc -l)

# No new snapshots exist. We cannot run HBT+
if [ $MAX_HBT_OUTPUT -eq $MAX_SNAPSHOT ]; then
  echo "HBT+ was done up to snapshot $(($MAX_HBT_OUTPUT - 1)). SWIFT outputs exist up to snapshot $(($MAX_SNAPSHOT - 1)). Cannot do more HBT+ now. Exiting."
  exit 1
fi

# We check how many ParticleSplit files exist
MAX_PARTICLE_SPLIT_OUTPUT=$(find $BASE_FOLDER/HBTplus/ParticleSplits -maxdepth 1 -name "particle_splits_????.hdf5" | wc -l)

echo "HBT+ was done up to snapshot $(($MAX_HBT_OUTPUT - 1)), and ParticleSplit information exists up to $(($MAX_PARTICLE_SPLIT_OUTPUT - 1)). SWIFT outputs exist up to snapshot $(($MAX_SNAPSHOT - 1))"

# This executes if we still need to generate the splitting of particles
if [ $MAX_PARTICLE_SPLIT_OUTPUT -ne $MAX_SNAPSHOT ]; then
  echo "Submitting splitting information from snapshots $MAX_PARTICLE_SPLIT_OUTPUT to $(($MAX_SNAPSHOT - 1))"
  JOB_ID_SPLITS=$(sbatch --parsable --output ${HBT_FOLDER}/logs/particle_splits/particle_splits.%A.%a.out --error ${HBT_FOLDER}/logs/particle_splits/particle_splits.%A.%a.err --array=$MAX_PARTICLE_SPLIT_OUTPUT-$(($MAX_SNAPSHOT - 1))%10 -J ${1} ./scripts/submit_particle_splits.sh $HBT_FOLDER/config.txt)

  # Submit an HBT job with a dependency on the splitting of particles
  echo "Submitting HBT+ dependency job, running from snapshots $MAX_HBT_OUTPUT to $(($MAX_SNAPSHOT - 1))"
  sbatch -J ${1} --dependency=afterok:$JOB_ID_SPLITS --output $HBT_FOLDER/logs/HBT.%j.out --error $HBT_FOLDER/logs/HBT.%J.err ./scripts/submit_HBT.sh $HBT_FOLDER/config.txt $MIN_SNAPSHOT $(($MAX_SNAPSHOT - 1))
else
  # We already have the splitting information, so we just need to run HBT without dependencies
  echo "Submitting HBT+ job without dependencies, running from snapshots $MAX_HBT_OUTPUT to $(($MAX_SNAPSHOT - 1))"
  sbatch -J ${1} --output $HBT_FOLDER/logs/HBT.%j.out --error $HBT_FOLDER/logs/HBT.%J.err ./scripts/submit_HBT.sh $HBT_FOLDER/config.txt $MAX_HBT_OUTPUT $(($MAX_SNAPSHOT - 1))
fi
