#ifndef SACRED_NONRECURSIVE_FILTER_GRADIENT_HPP_
#define SACRED_NONRECURSIVE_FILTER_GRADIENT_HPP_

#include "default_types.hpp"
#include "math.hpp"
#include "operator.hpp"
#include "tensor.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class NonrecursiveFilterGradient : public Operator<F> {
public:
  using tensor_type = typename default_tensor_type<F>::value;
  using tensors_type = typename default_tensors_type<F>::value;

  NonrecursiveFilterGradient(tensor_type &bias_gradient, tensor_type &filter, tensor_type &filter_gradient):
      bias_gradient_(bias_gradient), filter_(filter), filter_gradient_(filter_gradient) {}

  virtual ~NonrecursiveFilterGradient() = default;

  virtual void operator ()(const tensors_type &in, const tensors_type &out) override {
    // Filter derivatives:
    // ∂E/∂c_k = Σ_n ∂E/∂y_n * u_{n+k}
    // Note: my manual calculations figured ∂E/∂c_k = Σ_n ∂E/∂y_n * u_{n-k} but this disagrees
    //    with the dual number gradient verification. My mistake might be that
    //    ∂E/∂c_{-k} = Σ_n ∂E/∂y_n * u_{n-k} === ∂E/∂c_k = Σ_n ∂E/∂y_n * u_{n+k} is correct.
    // ∇_c E = ∇_y E ⁎ u
    math_.BackwardNarrowConvolve2(filter_gradient_, *out.at(0), *in.at(0), F(1.0), F(1.0));

    // Bottom derivatives:
    // ∂E/∂u_n = Σ_k ∂E/∂y_{n+k} c_k
    // ∇_u E = ∇_y E ⁎ c
    math_.BackwardWideConvolve2(*out.at(1), filter_, *in.at(0), F(1.0), F(1.0));

    // Bias derivatives:
    // ∂E/∂b_n = ∂E/∂y_n
    // ∇_b E = ∇_y E
    math_.Add(bias_gradient_, *in.at(0), F(1.0), F(1.0));
  }

private:
  tensor_type &bias_gradient_, &filter_, &filter_gradient_;
  Math<F> math_;
};

}  // namespace sacred

#endif  // SACRED_NONRECURSIVE_FILTER_GRADIENT_HPP_
