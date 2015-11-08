#ifndef SACRED_FULLY_CONNECTED_HPP_
#define SACRED_FULLY_CONNECTED_HPP_

#include "default_types.hpp"
#include "math.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class FullyConnectedGradient : public Operator<F> {
public:
  using tensor_type = typename default_tensor_type<F>::value;
  using tensors_type = typename default_tensors_type<F>::value;

  FullyConnectedGradient(tensor_type &bias_gradient, tensor_type &weight, tensor_type &weight_gradient):
      bias_gradient_(bias_gradient), weight_(weight), weight_gradient_(weight_gradient) {}

  virtual void operator ()(const tensors_type &in, const tensors_type &out) override {
    auto delta = in.at(0), x = out.at(0), x_gradient = out.at(1);

    // ∂E/∂W_ij = δ_i x_j
    math_.GeneralMatrixMultiplicationNormalTranspose(weight_gradient_, *delta, *x, F(1), F(1));

    // ∂E/∂b_i = δ_i
    math_.Add(bias_gradient_, *delta, F(1), F(1));

    // ∂E/∂x_j = δ_i W^i_j
    math_.GeneralMatrixMultiplicationTransposeNormal(*x_gradient, weight_, *delta, F(1), F(1));
  }

private:
  tensor_type &bias_gradient_, &weight_, &weight_gradient_;
  Math<F> math_;
};

}  // namespace sacred

#endif  // SACRED_FULLY_CONNECTED_HPP_
