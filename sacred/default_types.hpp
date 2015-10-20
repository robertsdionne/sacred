#ifndef SACRED_DEFAULT_TYPES_HPP_
#define SACRED_DEFAULT_TYPES_HPP_

#include <vector>

namespace sacred {

using std::vector;

using default_floating_point_type = float;
using default_integer_type = int;

template <typename F = default_floating_point_type>
struct default_storage_type {
  using value = vector<F>;
};

template <typename I = default_integer_type>
struct default_index_type {
  using value = vector<I>;
};

}  // namespace sacred

#endif  // SACRED_DEFAULT_TYPES_HPP_
