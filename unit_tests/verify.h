#ifndef VERIFY_H
#define VERIFY_H

#include <string>

void verify_failed(const std::string &message, const std::string &filename, const int line);

/*
  Abort with a message if condition x is not true.
  This is like assert() but is not disabled in Release mode.
*/
#define verify(X)                                                                                                      \
  if (!(X))                                                                                                            \
  verify_failed(#X, __FILE__, __LINE__)

#endif
