#ifndef SACRED_STRIDES_HPP_
#define SACRED_STRIDES_HPP_

#include <vector>

namespace sacred {

using std::vector;

namespace strides {

vector<int> CStyle(const vector<int> &shape) {
  auto stride = vector<int>(shape.size());
  auto product = 1;
  for (auto i = 0; i < shape.size(); ++i) {
    stride.at(shape.size() - 1 - i) = product;
    product *= shape.at(shape.size() - 1 - i);
  }
  return stride;
}

}  // namespace strides

}  // namespace sacred

#endif  // SACRED_STRIDES_HPP_
