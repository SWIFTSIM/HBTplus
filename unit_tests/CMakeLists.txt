include_directories(../src/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Helper functions available for use in all unit tests
add_library(testfuncs OBJECT
  verify.cpp
  make_test_subhalo.cpp
)

#
# Loop over unit tests to build and run. To add a new test:
#
# - put the code in a file tests/${TEST_NAME}.cpp
# - add ${TEST_NAME} to the foreach loop below
#
# Each test is an executable which should return zero on success or non-zero
# otherwise. Run tests with "make test". This does not use MPI so we can't
# test parallel code here.
#
# All HBT source files and library dependencies are linked into each
# test executable and the same macros are defined as for the main HBT
# executable, so tests need to be written to work with any combination
# of macros.
#
#
# First do tests which do not use MPI
#
foreach(TEST_NAME
    test_build_pos_vel
  )

  add_executable(${TEST_NAME} ${TEST_NAME}.cpp $<TARGET_OBJECTS:hbtfuncs> $<TARGET_OBJECTS:testfuncs>)
  target_link_libraries(${TEST_NAME} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES} ${GSL_LIBRARIES})
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

endforeach()

#
# Then run unit tests which do require MPI
# In this case we use the output from FindMPI to construct an mpiexec command.
#
set(HBT_MPI_TEST_NPROCS 4 CACHE STRING "Set the number of processors used in MPI unit tests")
foreach(TEST_NAME
    test_mpi
    test_exchange_counts
  )

  add_executable(${TEST_NAME} ${TEST_NAME}.cpp $<TARGET_OBJECTS:hbtfuncs> $<TARGET_OBJECTS:testfuncs>)
  target_link_libraries(${TEST_NAME} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${HDF5_C_LIBRARIES} ${HDF5_HL_LIBRARIES} ${GSL_LIBRARIES})
  add_test(NAME ${TEST_NAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${HBT_MPI_TEST_NPROCS} ${MPIEXEC_PREFLAGS} ${TEST_NAME} ${MPIEXEC_POSTFLAGS})
  
endforeach()
