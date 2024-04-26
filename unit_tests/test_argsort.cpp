#include <iostream>

#include "argsort.h"
#include "verify.h"

int main(int argc, char *argv[])
{

  std::vector<float> data{8, 4, 9, 3, 1, 7};
  std::vector<int> order = argsort<int, float>(data);

  for (int i = 1; i < data.size(); i += 1)
  {
    // std::cout << i << " " << order[i] << " " << data[order[i]] << std::endl;
    verify(data[order[i]] >= data[order[i - 1]]);
  }

  return 0;
}
