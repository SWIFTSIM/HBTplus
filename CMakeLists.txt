#
# Minimum cmake version:
# Setting C++ standard seems to require at least 3.9
#
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

#
# Set project name and language
#
project(hbtplus)
enable_language(C)
enable_language(CXX)
include(CTest)

#
# Set default build mode to Release
#
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (e.g. Debug, Release)" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

#
# Store library paths in executables
#
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#
# Put executables directly in build directory
#
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

#
# Specify which C++ standard we need
#
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#
# Find OpenMP flag for this compiler, if necessary
#
option(HBT_USE_OPENMP "Whether to enable OpenMP support" ON)
if(HBT_USE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_libraries(${OpenMP_CXX_FLAGS})
  else(OPENMP_FOUND)
    error("Unable to find OpenMP flag for this compiler")
  endif(OPENMP_FOUND)
else(HBT_USE_OPENMP)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # Suppress warnings about unrecognised '#pragma omp' from Intel compiler
    add_compile_options("-diag-disable 3180")
  endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
endif(HBT_USE_OPENMP)

#
# Find MPI libraries
#
find_package(MPI REQUIRED COMPONENTS C CXX)
include_directories(${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS})

#
# Find HDF5 library: we need the C and high level interfaces, and
# prefer to have MPI support if possible.
#
# Parallel HDF5 defines the preprocessor macro H5_HAVE_PARALLEL.
#
set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 COMPONENTS C HL REQUIRED)
include_directories(${HDF5_C_INCLUDE_DIRS})

#
# Find GSL library (optional)
#
option(HBT_USE_GSL "Enable to use GSL for eigenvalue decomposition of inertial tensors" OFF)
if(HBT_USE_GSL)
  find_package(GSL)
  if(GSL_FOUND)
    include_directories(${GSL_INCLUDE_DIRS})
    link_directories(${GSL_LIBRARY_DIRS})
    add_definitions(-DHAS_GSL)
  else(GSL_FOUND)
    error("Unable to find GSL library")
  endif(GSL_FOUND)
endif(HBT_USE_GSL)

#
# List of HBT source files.
# Headers are found automatically by cmake.
#
set(HBT_SRC 
  src/config_parser.cpp
  src/geometric_tree.cpp
  src/gravity_tree.cpp
  src/halo.cpp
  src/hdf_wrapper.cpp
  src/linkedlist_base.cpp
  src/linkedlist.cpp
  src/merger_tree.cpp
  src/mpi_wrapper.cpp
  src/mymath.cpp
  src/particle_exchanger.cpp
  src/snapshot.cpp
  src/snapshot_exchanger.cpp
  src/subhalo.cpp
  src/subhalo_merge.cpp
  src/subhalo_tracking.cpp
  src/subhalo_unbind.cpp
  src/subhalo_update_mostbound.cpp
  src/io/apostle_io.cpp
  src/io/gadget_group_io.cpp
  src/io/gadget_io.cpp
  src/io/swiftsim_io.cpp
  src/io/halo_io.cpp
  src/io/snapshot_io.cpp
  src/io/subhalo_io.cpp
  src/io/exchange_and_merge.cpp
  )

#
# Function for defining optional features
#
function(hbt_option NAME DEFINE DESCRIPTION DEFAULT)
  option(${NAME} ${DESCRIPTION} ${DEFAULT})
  if(${NAME})
    add_definitions(-D${DEFINE})
  endif(${NAME})
endfunction()

#
# Optional features:
#
#          CMake option name           Preprocessor macro      Description                                       Default
hbt_option(HBT_DM_ONLY                 DM_ONLY                 "Enable to save memory in DMONLY runs"            OFF)
hbt_option(HBT_INPUT_INT8              INPUT_INT8              "Enable to read 8 byte integers from input files" ON)
hbt_option(HBT_INT8                    HBT_INT8                "Enable to use 8 byte integers"                   ON)
hbt_option(HBT_REAL8                   HBT_REAL8               "Enable to use 8 byte reals"                      OFF)
hbt_option(HBT_UNSIGNED_LONG_ID_OUTPUT UNSIGNED_LONG_ID_OUTPUT "Enable to output IDs as unsigned long"           ON)
hbt_option(HBT_REAL4_VEL               HBT_REAL4_VEL           "Enable to store particle velocities as 4 byte reals" OFF)
hbt_option(HBT_REAL4_MASS              HBT_REAL4_MASS          "Enable to store particle masses as 4 byte reals" OFF)
hbt_option(HBT_UNBIND_WITH_THERMAL_ENERGY UNBIND_WITH_THERMAL_ENERGY "Enable use of thermal energy when unbinding" OFF)
hbt_option(HBT_CHECK_TRACER_INDEX      CHECK_TRACER_INDEX      "For debugging: check that tracer particle has the expected ID" OFF)

#
# Determine if we're going to output git revision info
#
set(HAVE_GIT 0)
find_package(Git)
if(GIT_FOUND)
  if(EXISTS ${CMAKE_SOURCE_DIR}/.git)
    set(HAVE_GIT 1)
  endif(EXISTS ${CMAKE_SOURCE_DIR}/.git)
endif(GIT_FOUND)
set(STORE_REVISION_INFO ${HAVE_GIT} CACHE BOOL "Whether to output git revision info")

#
# Generate a shell script which will run the git executable we found and generate the version info
# if any source file or the git repo changes.
#
if(STORE_REVISION_INFO)
  set(GIT_VERSION_INFO_FILE ${CMAKE_BINARY_DIR}/git_version_info.c)
  configure_file(${CMAKE_SOURCE_DIR}/git_version_info.sh.in ${CMAKE_BINARY_DIR}/git_version_info.sh @ONLY)
  add_custom_command(
    OUTPUT ${GIT_VERSION_INFO_FILE}
    COMMAND /bin/sh ${CMAKE_BINARY_DIR}/git_version_info.sh > ${GIT_VERSION_INFO_FILE}
    DEPENDS HBT.cpp ${HBT_SRC} ${CMAKE_BINARY_DIR}/git_version_info.sh CMakeLists.txt 
            ${CMAKE_SOURCE_DIR}/.git/HEAD ${CMAKE_SOURCE_DIR}/.git/index
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  add_definitions(-DHAVE_GIT)
endif(STORE_REVISION_INFO)
include_directories(${CMAKE_SOURCE_DIR})

#
# Make an object library with all HBT source files except the top level program.
# This makes a bundle of .o files which we can reuse when compiling unit tests.
#
add_library(hbtfuncs OBJECT ${HBT_SRC} ${GIT_VERSION_INFO_FILE})

#
# HBT executable
#
add_executable(HBT HBT.cpp $<TARGET_OBJECTS:hbtfuncs>)
target_link_libraries(HBT ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES} ${GSL_LIBRARIES})

#
# Hide some variables we usually don't need from the default view in the cmake GUI
#
mark_as_advanced(HDF5_DIR)
mark_as_advanced(HDF5_CXX_LIBRARY_dl)
mark_as_advanced(HDF5_CXX_LIBRARY_hdf5)
mark_as_advanced(HDF5_CXX_LIBRARY_hdf5_cpp)
mark_as_advanced(HDF5_CXX_LIBRARY_hdf5_hl)
mark_as_advanced(HDF5_CXX_LIBRARY_hdf5_hl_cpp)
mark_as_advanced(HDF5_CXX_LIBRARY_m)
mark_as_advanced(HDF5_CXX_LIBRARY_rt)
mark_as_advanced(HDF5_CXX_LIBRARY_z)
mark_as_advanced(CMAKE_INSTALL_PREFIX)

#
# Show C++ compiler flags in default view
#
mark_as_advanced(CLEAR CMAKE_CXX_FLAGS)
mark_as_advanced(CLEAR CMAKE_CXX_FLAGS_RELEASE)
mark_as_advanced(CLEAR CMAKE_CXX_FLAGS_DEBUG)

# Add unit tests directory
add_subdirectory(./test/unit_test)
