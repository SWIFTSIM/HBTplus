#include <cstdint>

/*
  Hash function for integer values.

  Intended to generate uniformly distributed output for any input so that
  we can hash a particle ID to determine which MPI rank to send it to,
  for example.

  See https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
*/

inline HBTInt HashInteger(HBTInt x) {
#ifdef HBT_INT8
  uint64_t y = x;
  y = (y ^ (y >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  y = (y ^ (y >> 27)) * UINT64_C(0x94d049bb133111eb);
  y = y ^ (y >> 31);
#else
  uint32_t y = x;
  y = ((y >> 16) ^ y) * UINT32_C(0x45d9f3b);
  y = ((y >> 16) ^ y) * UINT32_C(0x45d9f3b);
  y = (y >> 16) ^ y;
#endif
  return y;
}


/*
  Function to assign particle ID to an MPI rank based on hashing the ID
*/
inline int RankFromIdHash(HBTInt Id, int comm_size) {
  return std::abs(HashInteger(Id)) % comm_size;
}
