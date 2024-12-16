#include <mpi.h>
#include <time.h>
#include <errno.h>
#include <iostream>

#include "task_limited_section.h"
#include "verify.h"

/*
  Test code to limit number of tasks executing simultaneously.
*/
int main(int argc, char *argv[])
{

  MPI_Init(&argc, &argv);
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  int comm_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

  /* Skip this test if we're running on only one MPI rank */
  if(comm_size==1)return 0;

  /* Set up count of number of tasks executing */
  int *count;
  MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &count);
  *count = 0;
  MPI_Win win;
  MPI_Win_create(count, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  /* Split off one rank to maintain a count of tasks currently executing */
  int color = (comm_rank==0) ? 0 : 1;
  int key = comm_rank;
  MPI_Comm split_comm;
  MPI_Comm_split(MPI_COMM_WORLD, color, key, &split_comm);

  if(color == 1) {
    int split_comm_size = comm_size - 1;
    for(int max_nr_tasks = 1; max_nr_tasks <= split_comm_size; max_nr_tasks += 1) {

      TaskLimitedSection section(split_comm, max_nr_tasks);
      section.start();

      /* On starting, increment the counter */
      int start_count = -1;
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
      int to_add = 1;
      MPI_Get_accumulate(&to_add, 1, MPI_INT,
                         &start_count, 1, MPI_INT,
                         0, 0, 1, MPI_INT, MPI_SUM, win);
      MPI_Win_unlock(0, win);
      /* When we start, should have 0 to max_nr_tasks-1 other tasks running */
      verify(start_count >= 0);
      verify(start_count < max_nr_tasks);
      
      /* Sleep for a bit */
      struct timespec ts;
      ts.tv_sec = 0;
      ts.tv_nsec = 100 * 1000000; // 100 millisec
      int res;
      do {
        res = nanosleep(&ts, &ts);
      } while (res && errno == EINTR);

      /* On finishing, decrement the counter */
      int end_count = -1;
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
      to_add = -1;
      MPI_Get_accumulate(&to_add, 1, MPI_INT,
                         &end_count, 1, MPI_INT,
                         0, 0, 1, MPI_INT, MPI_SUM, win);
      MPI_Win_unlock(0, win);
      /* When we finish, should have 1 to max_nr_tasks tasks running (including our self) */
      verify(end_count > 0);
      verify(end_count <= max_nr_tasks);

      section.end();

      // Report maximum counts:
      // We should usually have start_count_max=max_nr_tasks-1 and end_count_max=max_nr_tasks,
      // although this is not guaranteed (e.g. if system is busy and some tasks are delayed).
      int start_count_max;
      MPI_Allreduce(&start_count, &start_count_max, 1, MPI_INT, MPI_MAX, split_comm);
      int end_count_max;
      MPI_Allreduce(&end_count, &end_count_max, 1, MPI_INT, MPI_MAX, split_comm);
      if(comm_rank==1)
        std::cout << "Max ranks = " << max_nr_tasks << ", max start count = " <<
          start_count_max << ", max end count = " << end_count_max << std::endl;  
    }
  }

  MPI_Win_free(&win);
  MPI_Free_mem(count);
  MPI_Finalize();

  return 0;
}
