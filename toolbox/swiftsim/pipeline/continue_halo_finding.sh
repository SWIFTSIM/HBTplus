#!/bin/bash

set -e

if [ "$#" -ne 1 ]
then
  echo "Usage: ./continue_halo_finding.sh <SIMULATION_NAME>"
  exit 1
fi

# Base folder of the simulation
BASE_FOLDER="${1}"
HBT_FOLDER=$BASE_FOLDER/HBTplus/

# There should be an HBT folder already there. If not, a path may be wrong.
if [ ! -d $HBT_FOLDER ]; then
    echo "No HBTplus folder found in provided path."
    exit 1
fi

# Check if the configuration file exists in the destination
if [ ! -f $HBT_FOLDER/config.txt ]; then
  echo "No configuration file exists at output folder. This script is only used to continue HBT+ runs. Exiting."
  exit 1
fi

# Where logs for particle splits and HBT will be saved (Should already exist)
HBT_LOGS_DIR="${HBT_FOLDER}/logs"
PARTICLE_SPLITS_LOGS_DIR="${HBT_LOGS_DIR}/particle_splits"

# We check how many SWIFT particle outputs have been done
MIN_SNAPSHOT=0
MAX_SNAPSHOT=$(find $BASE_FOLDER/$SNAPSHOT_SUBDIR/ -maxdepth 1 -name "colibre_????" | wc -l)

# We check how many HBT catalogues have been done
if [ ! -f $HBT_FOLDER/timing.log ]; then
  MAX_HBT_OUTPUT=0
else
  MAX_HBT_OUTPUT=$(tail -n 1 $HBT_FOLDER/timing.log | awk '{print $1}')
  MAX_HBT_OUTPUT=$((MAX_HBT_OUTPUT + 1))
fi

# No new snapshots exist. We cannot run HBT+
if [ $MAX_HBT_OUTPUT -eq $MAX_SNAPSHOT ]; then
  echo "HBT+ was done up to snapshot $(($MAX_HBT_OUTPUT - 1)). SWIFT outputs exist up to snapshot $(($MAX_SNAPSHOT - 1)). Cannot do more HBT+ now. Exiting."
  exit 1
fi

# We check how many ParticleSplit files exist
MAX_PARTICLE_SPLIT_OUTPUT=0
while true; do
    filename=$(printf "particle_splits_%04d.hdf5" "$MAX_PARTICLE_SPLIT_OUTPUT")
    if [ ! -f "${HBT_FOLDER}/ParticleSplits/$filename" ]; then
        break
    fi
    MAX_PARTICLE_SPLIT_OUTPUT=$((MAX_PARTICLE_SPLIT_OUTPUT + 1))
done

echo "HBT+ was done up to snapshot $(($MAX_HBT_OUTPUT - 1)), and ParticleSplit information exists up to $(($MAX_PARTICLE_SPLIT_OUTPUT - 1)). SWIFT outputs exist up to snapshot $(($MAX_SNAPSHOT - 1))"

# This executes if we still need to generate the splitting of particles
if [ $MAX_PARTICLE_SPLIT_OUTPUT -ne $MAX_SNAPSHOT ]; then
  echo "Submitting splitting information from snapshots $MAX_PARTICLE_SPLIT_OUTPUT to $(($MAX_SNAPSHOT - 1))"
  JOB_ID_SPLITS=$(sbatch --parsable \
    --output ${PARTICLE_SPLITS_LOGS_DIR}/particle_splits.%A.%a.out \
    --error ${PARTICLE_SPLITS_LOGS_DIR}/particle_splits.%A.%a.err \
    --array=$MAX_PARTICLE_SPLIT_OUTPUT-$(($MAX_SNAPSHOT - 1))%10 \
    -J "PS-${1}" \
    ${HBT_FOLDER}/submit_particle_splits.sh $HBT_FOLDER/config.txt)

  # Submit an HBT job with a dependency on the splitting of particles
  echo "Submitting HBT+ dependency job, running from snapshots $MAX_HBT_OUTPUT to $(($MAX_SNAPSHOT - 1))"
  sbatch -J "HBT-${1}" \
    --dependency=afterok:$JOB_ID_SPLITS \
    --output ${HBT_LOGS_DIR}/HBT.%j.out \
    --error ${HBT_LOGS_DIR}/HBT.%J.err \
    ${HBT_FOLDER}/submit_HBT.sh $HBT_FOLDER/config.txt $MAX_HBT_OUTPUT $(($MAX_SNAPSHOT - 1))
else
  # We already have the splitting information, so we just need to run HBT without dependencies
  echo "Submitting HBT+ job without dependencies, running from snapshots $MAX_HBT_OUTPUT to $(($MAX_SNAPSHOT - 1))"
  sbatch -J "HBT-${1}" \
    --output ${HBT_LOGS_DIR}/HBT.%j.out \
    --error ${HBT_LOGS_DIR}/HBT.%J.err \
    ${HBT_FOLDER}/submit_HBT.sh $HBT_FOLDER/config.txt $MAX_HBT_OUTPUT $(($MAX_SNAPSHOT - 1))
fi
