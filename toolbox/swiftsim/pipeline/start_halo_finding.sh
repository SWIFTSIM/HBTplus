#!/bin/bash

set -e

# Check if we have provided a path
if [ "$#" -ne 1 ]
then
  echo "Usage: ./start_halo_finding.sh <PATH_TO_SIMULATION>"
  exit 1
fi

# Base folder of the simulation
BASE_FOLDER=${1}

# There should be an HBT folder already there. If not, a path may be wrong.
if [ ! -d $BASE_FOLDER/HBTplus ]; then
    echo "No HBTplus folder found in provided path. Check whether path is correct, and if so, create an HBTplus folder first."
fi

# Where HBT outputs will be saved.
HBT_FOLDER=$BASE_FOLDER/HBTplus

# Determine whether the snapshots are saved in their own subdirectory (named snapshots) or not
SNAPSHOT_SUBDIR=""
if [ -d $BASE_FOLDER/snapshots ]; then
    SNAPSHOT_SUBDIR="snapshots"
fi

# Check if the configuration file exists in the destination
if [ ! -f $HBT_FOLDER/config.txt ]; then

  echo "Copying configuration file to $HBT_FOLDER/config.txt"

  # Get the number of snapshots we are expecting
  NUM_OUTPUTS="$(grep -v '^#' $BASE_FOLDER/output_list.txt | wc -l)"

  # Copy the template parameter file to the output folder
  cp ./template_config.txt $HBT_FOLDER/config.txt

  # Substitute the name of the simulation within the template
  sed -i "s@SIMULATION_PATH@${BASE_FOLDER}@g" $HBT_FOLDER/config.txt

  # Substitute the name of the subdirectory, if any
  sed -i "s@SNAPSHOT_SUBDIRECTORY@${SNAPSHOT_SUBDIR}@g" $HBT_FOLDER/config.txt

  # Substitute the number of outputs within the HBT template
  sed -i "s@MAXIMUM_SNAPSHOT@$(($NUM_OUTPUTS - 1))@g" $HBT_FOLDER/config.txt

else
  echo "Configuration file already exists. Exiting to prevent accidental overwrites."
  exit 1
fi

# Where logs for particle splits and HBT will be saved
mkdir -p $HBT_FOLDER/logs/particle_splits/

# We now check how many COLIBRE snapshots have been done at the time of submission
MIN_SNAPSHOT=0
MAX_SNAPSHOT=$(find $BASE_FOLDER/$SNAPSHOT_SUBDIR/ -maxdepth 1 -name "colibre_????" | wc -l)

# Abort if no snapshots are found
if [ $MAX_SNAPSHOT -eq  0 ]; then
  echo "Could not find COLIBRE snapshots in the specified directory. Exiting."
  exit 1
fi

echo "Submitting an HBT+ job running from snapshots $MIN_SNAPSHOT to $(($MAX_SNAPSHOT - 1))"

# We first generate the splitting of particles 
JOB_ID_SPLITS=$(sbatch --parsable --output ${HBT_FOLDER}/logs/particle_splits/particle_splits.%A.%a.out --error ${HBT_FOLDER}/logs/particle_splits/particle_splits.%A.%a.err --array=$MIN_SNAPSHOT-$(($MAX_SNAPSHOT - 1))%10 -J ${1} ./scripts/submit_particle_splits.sh $HBT_FOLDER/config.txt)

# Submit an HBT job with a dependency on the splitting of particles
sbatch -J ${1} --dependency=afterok:$JOB_ID_SPLITS --output $HBT_FOLDER/logs/HBT.%j.out --error $HBT_FOLDER/logs/HBT.%J.err ./scripts/submit_HBT.sh $HBT_FOLDER/config.txt $MIN_SNAPSHOT $(($MAX_SNAPSHOT - 1))
