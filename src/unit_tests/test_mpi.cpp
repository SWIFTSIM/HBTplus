#include <mpi.h>

// Test that we can compile and run an MPI executable
int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  MPI_Finalize();

  return 0;
}
