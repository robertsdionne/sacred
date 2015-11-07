#ifndef SACRED_NONRECURSIVE_FILTER_HPP_
#define SACRED_NONRECURSIVE_FILTER_HPP_

#include "default_types.hpp"
#include "math.hpp"
#include "operator.hpp"
#include "tensor.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class NonrecursiveFilter : public Operator<F> {
public:
  using tensor_type = typename default_tensor_type<F>::value;
  using tensors_type = typename default_tensors_type<F>::value;

  NonrecursiveFilter(tensor_type &bias, tensor_type &filter) : bias_(bias), filter_(filter) {}

  virtual ~NonrecursiveFilter() = default;

  virtual void operator ()(const tensors_type &in, const tensors_type &out) override {
    // y = c ⁎ u + b
    // y_n = Σ_k c_k * u_{n-k} + b_n
    math_.NarrowConvolve2(*out.at(0), filter_, *in.at(0), F(1.0), F(1.0));
    math_.Add(*out.at(0), bias_, F(1.0), F(1.0));
  }

private:
  tensor_type &bias_, &filter_;
  Math<F> math_;
};

}  // namespace sacred

#endif  // SACRED_NONRECURSIVE_FILTER_HPP_
