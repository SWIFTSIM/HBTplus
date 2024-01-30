# Load modules used for compiling
module purge
module load gnu_comp/11.1.0 hdf5/1.12.0 openmpi/4.1.1 cmake/3.25.1

# Location of executable
rm -fr build # In case an older version is already present
mkdir build

# Create cmake file and configuration
cmake -B$PWD/build -S"$PWD/../.." -D HBT_USE_OPENMP=ON -D HBT_UNSIGNED_LONG_ID_OUTPUT=OFF

# Compile
cd $PWD/build
make -j 4