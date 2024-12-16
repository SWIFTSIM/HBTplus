#include <mpi.h>
#include <assert.h>
#include <cstdlib>

/*
  Class to limit the number of MPI ranks executing a piece of code
  simultaneously. Used to implement the MaxConcurrentIO option.

  Example usage:

  TaskLimitedSection section(MPI_COMM_WORLD, HBTConfig.MaxConcurrentIO);
  section.start();
  ...
  (I/O code goes here!)
  ...
  section.end();

*/
class TaskLimitedSection {

private:

  int max_nr_tasks;
  MPI_Comm comm;
  MPI_Win win;
  int *buffer;
  int controller_rank;
  MPI_Request controller_rank_request;
  int order;
  
  const int CONTROLLER_RANK_TAG = 0;
  const int GO_TAG = 1;
  const int COMPLETION_TAG = 2;

public:

  TaskLimitedSection(MPI_Comm comm, const int max_nr_tasks) {

    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    // Renumber ranks so we're not just allowing the first N to run initially -
    // ideally we want to have the active ranks spread over all compute nodes.
    int block_size = max_nr_tasks;
    int position_in_block = comm_rank % block_size;
    int block_index = comm_rank / block_size;
    int nr_blocks = comm_size / max_nr_tasks;
    if(comm_size % max_nr_tasks != 0)nr_blocks += 1;
    assert(block_size*block_index+position_in_block == comm_rank);
    order = position_in_block * nr_blocks + block_index;

    // Create the reordered communicator
    MPI_Comm_split(comm, 0, order, &(this->comm));
    this->max_nr_tasks = max_nr_tasks;
  }

  ~TaskLimitedSection() {
    MPI_Comm_free(&comm);
  }
  
  void start() {
  
    /* Get rank and number of ranks */
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    /* If all ranks are allowed to run there's nothing to do */
    if(max_nr_tasks >= comm_size)return;

    /* Allocate and init counter for RMA */
    MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &buffer);
    *buffer = 0;
    MPI_Win_create(buffer, sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &win);

    /* Post a receive to get controller task's rank (will be first rank to finish) */
    MPI_Irecv(&controller_rank, 1, MPI_INT, MPI_ANY_SOURCE,
              CONTROLLER_RANK_TAG, comm, &controller_rank_request);
    
    /* The first max_nr_tasks ranks can proceed immediately */
    if(comm_rank < max_nr_tasks)return;

    /* Others need to wait for a message to proceed */
    int go;
    MPI_Recv(&go, 1, MPI_INT, MPI_ANY_SOURCE, GO_TAG, comm, MPI_STATUS_IGNORE);
    
  }
  
  void end() {

    /* Get rank and number of ranks */
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    /* If all ranks are allowed to run there's nothing to do */
    if(max_nr_tasks >= comm_size)return;

    MPI_Request *request = (MPI_Request *) malloc(sizeof(MPI_Request)*comm_size);
  
    /*
      Check if we're the first task to reach the end of the section:
      We do this by doing an atomic fetch and increment on the count of
      the number of ranks that have finished. If the count is zero we're
      the first and will become responsible for signalling other ranks
      to proceed.
      
      We only need to check the completion count for the first max_nr_tasks
      ranks, because others can't start until another rank finishes so they
      can't be first to finish.
    */
    int completion_count = 0;
    if(comm_rank < max_nr_tasks) {
      /* We're one of the ranks that started immediately, so we might be first
         to complete */
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, comm_size-1, 0, win);
      int to_add = 1;
      MPI_Get_accumulate(&to_add, 1, MPI_INT,
                         &completion_count, 1, MPI_INT,
                         comm_size-1, 0, 1, MPI_INT, MPI_SUM, win);
      MPI_Win_unlock(comm_size-1, win);
    } else {
      /* We aren't in the initial batch of max_nr_tasks so we can't be first to complete.
         Skip the get_accumulate so we're not waiting for the last rank to respond
         when it might be busy in non-MPI code. */
      completion_count = 1;
    }
    if(completion_count == 0) {
      
      /* This task is the first to reach the end of the section, so tell everyone */
      for(int dest=0; dest<comm_size; dest+=1)
        MPI_Isend(&comm_rank, 1, MPI_INT, dest, CONTROLLER_RANK_TAG, comm, request+dest);

      /* Then we need to wait for others to finish and send go signals as necessary */
      int nr_left = comm_size - max_nr_tasks;
      for(int i=0; i<comm_size; i+=1) {
        
        /* Wait for a completion message (but not on first iteration, because we don't send to self) */
        if(i > 0) {
          int done;
          MPI_Status status;
          MPI_Recv(&done, 1, MPI_INT, MPI_ANY_SOURCE, COMPLETION_TAG, comm, &status);
        }
        
        /* If there are tasks still waiting, send the next go signal */
        if(nr_left > 0) {
          int dest = comm_size - nr_left;
          int go = 1;
          MPI_Send(&go, 1, MPI_INT, dest, GO_TAG, comm);
          nr_left -= 1;
        }
      }
    }

    /* Make sure we've received the controller rank */
    MPI_Wait(&(controller_rank_request), MPI_STATUS_IGNORE);
  
    /* Send completion message if we're not the controller */
    if(completion_count > 0) {
      int complete = 1;
      MPI_Send(&complete, 1, MPI_INT, controller_rank, COMPLETION_TAG, comm);
    }

    /* Make sure all sends from the controller completed */
    if(completion_count==0)
      MPI_Waitall(comm_size, request, MPI_STATUSES_IGNORE);
  
    /* Tidy up */
    free(request);
    MPI_Win_free(&win);
    MPI_Free_mem(buffer);
  }
  
};
