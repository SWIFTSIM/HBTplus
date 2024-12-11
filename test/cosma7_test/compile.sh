# Load modules used for compiling
module purge
module load gnu_comp/13.1.0 hdf5/1.12.2 openmpi/4.1.4

# Location of executable
rm -fr build # In case an older version is already present
mkdir build

# Create cmake file and configuration
cmake -B$PWD/build -S"$PWD/../.." -D HBT_USE_OPENMP=ON -D HBT_UNSIGNED_LONG_ID_OUTPUT=OFF -D CMAKE_BUILD_TYPE=Debug

# Compile
cd $PWD/build
make -j 4
