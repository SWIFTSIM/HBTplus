#ifndef ARGSORT_H
#define ARGSORT_H

#include <vector>
#include <numeric>
#include <algorithm>

/*
  Perform an indirect sort. This makes an index array to access the supplied
  array in sorted order, similar to numpy.argsort.

  Example usage:
  
  std::vector<float> data{8, 4, 9, 3, 1, 7};
  std::vector<int> order = argsort<int,float>(data);

*/
template <typename index_t, typename value_t> std::vector<index_t> argsort(std::vector<value_t> values) {

  std::vector<index_t> index(values.size());
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(), [&values](index_t a, index_t b) { return values[a] < values[b]; } );
  return index;
}

#endif
