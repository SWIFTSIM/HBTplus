#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdio.h>
#include <string>

int main()
{
  std::string fmt = "%d/%s.0";
  char buf[1024];
  sprintf(buf, fmt.c_str(), 3, "hi");
  std::cout << buf;

  return 0;
}