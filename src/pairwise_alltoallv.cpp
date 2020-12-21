#include <mpi.h>

int Pairwise_Alltoallv(const void *sendbuf, const int *sendcounts,
                       const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                       const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  
  int ptask = 0;
  while(1<<ptask < comm_size)
    ptask += 1;

  int send_type_size;
  MPI_Type_size(sendtype, &send_type_size);
  int recv_type_size;
  MPI_Type_size(recvtype, &recv_type_size);
  
  for(int ngrp=0; ngrp < (1<<ptask); ngrp+=1)
    {
      int rank = comm_rank ^ ngrp;
      if(rank < comm_size)
        {

          char *sendptr = ((char *) sendbuf) + 
            ((size_t) sdispls[rank]) * ((size_t) send_type_size);
          char *recvptr = ((char *) recvbuf) + 
            ((size_t) rdispls[rank]) * ((size_t) recv_type_size);

          int sendcount = sendcounts[rank];
          int recvcount = recvcounts[rank];

          while (sendcount > 0 || recvcount > 0)
            {
              size_t max_bytes = 1 << 30;
              size_t max_num_send = max_bytes / send_type_size;
              size_t max_num_recv = max_bytes / recv_type_size;
              size_t num_send = sendcount <= max_num_send ? sendcount : max_num_send;
              size_t num_recv = recvcount <= max_num_recv ? recvcount : max_num_recv;

              if(sendcount > 0 && recvcount > 0) 
                {
                  MPI_Sendrecv(sendptr, (int) num_send, sendtype, rank, 0,
                               recvptr, (int) num_recv, recvtype, rank, 0,
                               comm, MPI_STATUS_IGNORE);
                } 
              else if(sendcount > 0)
                {
                  MPI_Send(sendptr, (int) num_send, sendtype, rank, 0,
                           comm);
                }
              else
                {
                  MPI_Recv(recvptr, (int) num_recv, recvtype, rank, 0,
                           comm, MPI_STATUS_IGNORE);                  
                }

              sendptr   += send_type_size * num_send;
              sendcount -= num_send;

              recvptr   += recv_type_size * num_recv;
              recvcount -= num_recv;
            }
        }
    }  
  return 0;
}
