#ifndef SACRED_CHECKS_HPP_
#define SACRED_CHECKS_HPP_

#include <cstdlib>
#include <iostream>
#include <string>

#define CHECK_STATE(state) \
  sacred::CheckState(#state, state, __LINE__, __FILE__)

namespace sacred {

using std::string;

inline void Fail(const string &message, int line, const string &file);

inline void CheckState(const string &message, bool state, int line, const string &file) {
  if (!state) {
    Fail(message + u8" is violated", line, file);
  }
}

inline void Fail(const string &message, int line, const string &file) {
  std::cerr << u8"ERROR: " << message << u8" on line " << line << u8" of file " << file << std::endl;
  exit(1);
}

}  // namespace sacred

#endif  // SACRED_CHECKS_HPP_
