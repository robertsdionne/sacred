#ifndef SACRED_FULLY_CONNECTED_GRADIENT_HPP_
#define SACRED_FULLY_CONNECTED_GRADIENT_HPP_

#include "default_types.hpp"
#include "math.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class FullyConnectedGradient : public Operator<F> {
public:
  USING_TENSOR_TYPES(F);

  FullyConnectedGradient(tensor_type &bias_gradient, tensor_type &weight, tensor_type &weight_gradient):
      bias_gradient_(bias_gradient), weight_(weight), weight_gradient_(weight_gradient) {}

  void operator ()(const tensor_type &delta, const tensor_type &x, tensor_type &x_gradient) {
    // ∂E/∂W_ij = δ_i x_j
    for (auto i = 0; i < delta.shape().at(0); ++i) {
      for (auto q = 0; q < delta.shape().at(1); ++q) {
        for (auto r = 0; r < delta.shape().at(2); ++r) {
          for (auto j = 0; j < x.shape().at(1); ++j) {
            for (auto k = 0; k < x.shape().at(2); ++k) {
              weight_gradient_.set({q, r, j, k},
                  F(1) * weight_gradient_.at({q, r, j, k}) + F(1) * delta.at({i, q, r}) * x.at({i, j, k}));
            }
          }
        }
      }
    }

    // ∂E/∂b_i = δ_i
    for (auto i = 0; i < delta.shape().at(0); ++i) {
      for (auto q = 0; q < delta.shape().at(1); ++q) {
        for (auto r = 0; r < delta.shape().at(2); ++r) {
          bias_gradient_.set({q, r}, F(1) * bias_gradient_.at({q, r}) + F(1) * delta.at({i, q, r}));
        }
      }
    }

    // ∂E/∂x_j = δ_i W^i_j
    // math_.GeneralMatrixMultiplicationTransposeNormal(x_gradient, weight_, delta, F(1), F(1));
    for (auto i = 0; i < delta.shape().at(0); ++i) {
      for (auto j = 0; j < x.shape().at(1); ++j) {
        for (auto k = 0; k < x.shape().at(2); ++k) {
          F current_output = F(1) * x_gradient.at({i, j, k});
          for (auto q = 0; q < delta.shape().at(1); ++q) {
            for (auto r = 0; r < delta.shape().at(2); ++r) {
              current_output += F(1) * weight_.at({q, r, j, k}) * delta.at({i, q, r});
            }
          }
          x_gradient.set({i, j, k}, current_output);
        }
      }
    }
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto delta = in.at(0), x = in.at(1);
    auto x_gradient = out.at(0);
    operator ()(*delta, *x, *x_gradient);
  }

private:
  tensor_type &bias_gradient_, &weight_, &weight_gradient_;
  Math<F> math_;
};

}  // namespace sacred

#endif  // SACRED_FULLY_CONNECTED_GRADIENT_HPP_
