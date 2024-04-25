#ifndef REORDER_H
#define REORDER_H

#include <vector>
#include <cassert>

/*
  Reorder a vector using a sorting index
*/
template <typename value_t, typename index_t>
void reorder(std::vector<value_t> &arr, const std::vector<index_t> &order) {

  assert(arr.size()==order.size());
  
  std::vector<value_t> arr_sorted(order.size());
#pragma omp parallel for schedule(static, 10*1024)
  for(index_t i=0; i<order.size(); i+=1) {
    arr_sorted[i] = arr[order[i]];
  }
  arr.swap(arr_sorted);
}

#endif
