#ifndef PAIRWISE_ALLTOALLV_H
#define PAIRWISE_ALLTOALLV_H

#include <mpi.h>

/*
  MPI_Alltoallv which can handle large counts.
  
  Template parameter T should be an integer type large enough to store the largest
  count on any MPI rank.
*/
template<typename T>
int Pairwise_Alltoallv(const void *sendbuf, const T *sendcounts, const T *sdispls, MPI_Datatype sendtype,
                       void *recvbuf, const T *recvcounts, const T *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  int ptask = 0;
  while (1 << ptask < comm_size)
    ptask += 1;

  MPI_Aint lower_bound, extent;
  MPI_Type_get_extent(sendtype, &lower_bound, &extent);
  int send_type_size = extent;
  MPI_Type_get_extent(recvtype, &lower_bound, &extent);
  int recv_type_size = extent;

  for (int ngrp = 0; ngrp < (1 << ptask); ngrp += 1)
  {
    int rank = comm_rank ^ ngrp;
    if (rank < comm_size)
    {

      char *sendptr = ((char *)sendbuf) + ((size_t)sdispls[rank]) * ((size_t)send_type_size);
      char *recvptr = ((char *)recvbuf) + ((size_t)rdispls[rank]) * ((size_t)recv_type_size);

      size_t sendcount = (size_t) sendcounts[rank];
      size_t recvcount = (size_t) recvcounts[rank];

      while (sendcount > 0 || recvcount > 0)
      {
        size_t max_bytes = 1 << 30;
        size_t max_num_send = max_bytes / send_type_size;
        size_t max_num_recv = max_bytes / recv_type_size;
        size_t num_send = sendcount <= max_num_send ? sendcount : max_num_send;
        size_t num_recv = recvcount <= max_num_recv ? recvcount : max_num_recv;

        if (sendcount > 0 && recvcount > 0)
        {
          MPI_Sendrecv(sendptr, (int)num_send, sendtype, rank, 0, recvptr, (int)num_recv, recvtype, rank, 0, comm,
                       MPI_STATUS_IGNORE);
        }
        else if (sendcount > 0)
        {
          MPI_Send(sendptr, (int)num_send, sendtype, rank, 0, comm);
        }
        else
        {
          MPI_Recv(recvptr, (int)num_recv, recvtype, rank, 0, comm, MPI_STATUS_IGNORE);
        }

        sendptr += send_type_size * num_send;
        sendcount -= num_send;

        recvptr += recv_type_size * num_recv;
        recvcount -= num_recv;
      }
    }
  }
  return 0;
}

/*
  This version accepts vector arguments. Recvbuf must already be large enough for the result.
*/
template<typename T, typename U, typename V>
int Pairwise_Alltoallv(const std::vector<U> &sendbuf, const std::vector<T> &sendcounts, const std::vector<T> &sdispls, MPI_Datatype sendtype,
                       std::vector<V> &recvbuf, const std::vector<T> &recvcounts, const std::vector<T> &rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  return Pairwise_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), sendtype,
                            recvbuf.data(), recvcounts.data(), rdispls.data(), recvtype, comm);
}

/*
  Given number of elements to send to each rank, compute send displacements and receive counts and displacements
*/
template<typename T>
void ExchangeCounts(const std::vector<T> &sendcounts, std::vector<T> &sdispls,
                    std::vector<T> &recvcounts, std::vector<T> &rdispls,
                    MPI_Comm comm) {

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  // Exchange counts
  std::vector<long long> sendcounts_ll(comm_size);
  std::vector<long long> recvcounts_ll(comm_size);
  for(int i=0; i<comm_size; i+=1)
    sendcounts_ll[i] = static_cast<long long>(sendcounts[i]);
  MPI_Alltoall(sendcounts_ll.data(), 1, MPI_LONG_LONG, recvcounts_ll.data(), 1, MPI_LONG_LONG, comm);

  // Ensure output vectors are the right size
  sdispls.resize(comm_size);
  recvcounts.resize(comm_size);
  rdispls.resize(comm_size);
  
  // Populate output vectors
  for(int i=0; i<comm_size; i+=1)
    recvcounts[i] = static_cast<T>(recvcounts_ll[i]);
  sdispls[0] = 0;
  rdispls[0] = 0;
  for(int i=1; i<comm_size; i+=1) {
    sdispls[i] = sdispls[i-1] + sendcounts[i-1];
    rdispls[i] = rdispls[i-1] + recvcounts[i-1];    
  }
}

#endif
