#ifndef SACRED_OPERATOR_HPP_
#define SACRED_OPERATOR_HPP_

#include "default_types.hpp"
#include "tensor.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class Operator {
public:
  USING_TENSOR_TYPES(F);

  virtual ~Operator() = default;

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) = 0;
};

}  // namespace sacred

#endif  // SACRED_OPERATOR_HPP_
