#include <string>
#include <iostream>

#include "verify.h"

void verify_failed(std::string message, std::string filename, int line) {

  std::cerr << "Test failed: " << message << " at " << filename << ": " << line << std::endl;  
  exit(1);
}
