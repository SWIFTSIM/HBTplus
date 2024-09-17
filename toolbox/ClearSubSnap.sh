#!/usr/bin/bash
set -e

# Check if we have provided a path
if [ "$#" -ne 1 ]
then
  echo "Usage: ./ClearSubSnap.sh <PATH_TO_HBT_DIR>"
  exit 1
fi

# Check whether there is a timing that we can use to
# determine which snapshots have been done
BASE_PATH=${1}
if [ ! -f $BASE_PATH/timing.log ]; then
  echo "No timing.log found"
  exit 1
fi

# We make sure not to delete the last snapshot we analysed, in case
# we need to use it as a restart.
MAX_SNAP_TO_DELETE=$(tail -n 1 $BASE_PATH/timing.log | awk '{print $1}')
MAX_SNAP_TO_DELETE=$((MAX_SNAP_TO_DELETE - 1))

if [ $MAX_SNAP_TO_DELETE -lt 3 ]; then
  echo "Less than 3 snapshots to delete"
  exit 1
fi
echo "Maximum snap to delete: ${MAX_SNAP_TO_DELETE}"

# Delete
for ((i=0; i<=$MAX_SNAP_TO_DELETE; i++)); do
  SNAP_NR=$(printf "%03d" $i)
  rm $BASE_PATH/${SNAP_NR}/SrcSnap*
done
