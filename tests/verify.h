#include <string>

void verify_failed(std::string message, std::string filename, int line);

/*
  Abort with a message if condition x is not true.
  This is like assert() but is not disabled in Release mode.
*/
#define verify(X) if (!(X)) verify_failed( #X , __FILE__ , __LINE__)
