#ifndef SACRED_DEFAULT_TYPES_HPP_
#define SACRED_DEFAULT_TYPES_HPP_

#include <limits>
#include <vector>

#include "dual.hpp"

namespace sacred {

using std::numeric_limits;
using std::vector;

#define USING_TENSOR_TYPES(F) \
  using index_type = typename default_index_type<>::value; \
  using tensor_type = typename default_tensor_type<F>::value; \
  using tensors_type = typename default_tensors_type<F>::value; \
  using tensors_const_type = typename default_tensors_const_type<F>::value;

using default_floating_point_type = float;
using default_integer_type = int;

template <typename F>
struct epsilon {
  static const F value;
};

template <typename F>
const F epsilon<F>::value = numeric_limits<F>::epsilon();

template <>
struct epsilon<Dual> {
  static const Dual value;
};

const Dual epsilon<Dual>::value = Dual(numeric_limits<float>::epsilon());

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
  using const_value = vector<const typename default_tensor_type<F, I>::value *>;
};

template <typename F = default_floating_point_type, typename I = default_integer_type>
struct default_tensors_const_type {
  using value = vector<const typename default_tensor_type<F, I>::value *>;
};

}  // namespace sacred

#endif  // SACRED_DEFAULT_TYPES_HPP_
