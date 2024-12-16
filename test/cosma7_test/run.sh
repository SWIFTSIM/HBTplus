#!/bin/bash

if [[ $(hostname) != *"login7"* ]]
then
    echo "Trying to run test in a non-COSMA7 login node. Please switch to COSMA7."
else
    OUTPUTDIR=${1:-$PWD\/test_output}
    rm -rf $OUTPUTDIR # Remove to prevent partially overwritting previous tests.
    bash compile.sh
    mkdir -p $OUTPUTDIR/logs
    sbatch --export=OUTPUTDIR=$OUTPUTDIR --output $OUTPUTDIR/logs/output.out --error $OUTPUTDIR/logs/error.err submit.sh
fi
