#!/bin/bash

if [[ $(hostname) != *"login7"* ]]
then
    echo "Trying to run test in a non-COSMA7 login node. Please switch to COSMA7."
else
    rm -rf test_output # Remove to prevent partially overwritting previous tests.
    bash compile.sh
    sbatch submit_test.sh
fi