#ifndef SACRED_OPERATOR_HPP_
#define SACRED_OPERATOR_HPP_

#include "default_types.hpp"
#include "tensor.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class Operator {
public:
  using tensor_type = typename default_tensor_type<F>::value;
  using tensors_type = typename default_tensors_type<F>::value;

  virtual ~Operator() = default;

  virtual void Run(const tensors_type &in, const tensors_type &out) = 0;
};

}  // namespace sacred

#endif  // SACRED_OPERATOR_HPP_
