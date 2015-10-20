#ifndef SACRED_STRIDES_HPP_
#define SACRED_STRIDES_HPP_

#include <vector>

#include "default_types.hpp"

namespace sacred {

using std::vector;

namespace strides {

template <typename I = default_integer_type>
vector<I> CStyle(const vector<I> &shape) {
  auto stride = vector<I>(shape.size());
  auto product = I(1);
  for (auto i = I(0); i < shape.size(); ++i) {
    stride.at(shape.size() - 1 - i) = product;
    product *= shape.at(shape.size() - 1 - i);
  }
  return stride;
}

}  // namespace strides

}  // namespace sacred

#endif  // SACRED_STRIDES_HPP_
