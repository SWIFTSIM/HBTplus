# Load modules used for compiling
module purge
module load gnu_comp/11.1.0 hdf5/1.12.0 openmpi/4.1.1 cmake/3.25.1

# Location of executable
rm -fr build # In case an older version is already present
mkdir build

# Switch to top directory and clean it
cd ../../
make clean

# Create cmake file and configuration
cmake -B$PWD/configs/test/build -S$PWD -D HBT_USE_OPENMP=ON -D HBT_UNSIGNED_LONG_ID_OUTPUT=OFF

# Compile
cd $PWD/configs/test/build
make -j 4