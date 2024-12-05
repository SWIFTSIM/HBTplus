#include <string>
#include <iostream>

#include "verify.h"

void verify_failed(const std::string &message, const std::string &filename, const int line)
{

  std::cerr << "Test failed: " << message << " at " << filename << ": " << line << std::endl;
  exit(1);
}
