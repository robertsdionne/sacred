#ifndef SACRED_TESTING_HPP_
#define SACRED_TESTING_HPP_

#include <algorithm>
#include <iterator>
#include <ostream>
#include <vector>

namespace sacred {

using namespace std;

template <typename F> ostream &operator <<(ostream &out, const vector<F> &v) {
  out << "{";
  if (!v.empty()) {
    copy(begin(v), end(v) - 1, ostream_iterator<F>(out, ", "));
    copy(end(v) - 1, end(v), ostream_iterator<F>(out));
  }
  out << "}";
  return out;
}

}

#endif  // SACRED_TESTING_HPP_
