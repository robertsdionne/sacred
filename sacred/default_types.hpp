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

template <typename F, typename I>
class Tensor;

template <typename F = default_floating_point_type, typename I = default_integer_type>
struct default_tensor_type {
  using value = Tensor<F, I>;
};

template <typename F = default_floating_point_type, typename I = default_integer_type>
struct default_tensors_type {
  using value = vector<typename default_tensor_type<F, I>::value *>;
};

}  // namespace sacred

#endif  // SACRED_DEFAULT_TYPES_HPP_
