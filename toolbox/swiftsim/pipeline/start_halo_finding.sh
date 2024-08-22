#!/bin/bash

set -e

# Check if we have provided a path
if [ "$#" -ne 1 ]
then
  echo "Usage: ./start_halo_finding.sh <PATH_TO_SIMULATION>"
  exit 1
fi

# Base folder of the simulation
BASE_PATH=${1}
HBT_FOLDER=$BASE_PATH/HBTplus

# There should be an HBT folder already there. If not, a path may be wrong.
if [ ! -d $HBT_FOLDER ]; then
    echo "No HBTplus folder found in provided path. Check whether path is correct, and if so, create an HBTplus folder first."
fi

# Where logs for particle splits and HBT will be saved
HBT_LOGS_DIR="${HBT_FOLDER}/logs"
PARTICLE_SPLITS_LOGS_DIR="${HBT_LOGS_DIR}/particle_splits"
mkdir -p $PARTICLE_SPLITS_LOGS_DIR

# Set permissions
chmod ug+rw $HBT_LOGS_DIR
chmod ug+rw $PARTICLE_SPLITS_LOGS_DIR
setfacl -d -m g:dp004:rwX $HBT_FOLDER

# Where particle splits will be saved
mkdir "${HBT_FOLDER}/ParticleSplits"

# Copy over the configuration file with the correct paths
(cd ./templates/;. generate_config.sh $BASE_PATH $HBT_FOLDER)

# We now check how many COLIBRE snapshots have been done at the time of submission
MIN_SNAPSHOT=0
MAX_SNAPSHOT=$(find $BASE_PATH/$SNAPSHOT_SUBDIR/ -maxdepth 1 -name "colibre_????" | wc -l)

# Abort if no snapshots are found
if [ $MAX_SNAPSHOT -eq  0 ]; then
  echo "Could not find COLIBRE snapshots in the specified directory. Exiting."
  exit 1
fi

echo "Submitting an HBT+ job running from snapshots $MIN_SNAPSHOT to $(($MAX_SNAPSHOT - 1))"

# Copy the submission scripts into the HBT folders
cp ./submission_scripts/submit_HBT.sh $HBT_FOLDER
cp ./submission_scripts/submit_particle_splits.sh $HBT_FOLDER

# Replace the current PWD in those submission scripts, so that they have the global path of the HBT
sed -i "s@CURRENT_PWD@${PWD}@g" $HBT_FOLDER/submit_HBT.sh
sed -i "s@CURRENT_PWD@${PWD}@g" $HBT_FOLDER/submit_particle_splits.sh

# We first generate the splitting of particles 
JOB_ID_SPLITS=$(sbatch --parsable \
  --output ${PARTICLE_SPLITS_LOGS_DIR}/particle_splits.%A.%a.out \
  --error ${PARTICLE_SPLITS_LOGS_DIR}/particle_splits.%A.%a.err \
  --array=$MIN_SNAPSHOT-$(($MAX_SNAPSHOT - 1))%10 \
  -J "PS-${1}" \
  $HBT_FOLDER/submit_particle_splits.sh $HBT_FOLDER/config.txt)

# Submit an HBT job with a dependency on the splitting of particles
sbatch -J "HBT-${1}" \
  --dependency=afterok:$JOB_ID_SPLITS \
  --output ${HBT_LOGS_DIR}/HBT.%j.out \
  --error ${HBT_LOGS_DIR}/HBT.%J.err \
  $HBT_FOLDER/submit_HBT.sh $HBT_FOLDER/config.txt $MIN_SNAPSHOT $(($MAX_SNAPSHOT - 1))
