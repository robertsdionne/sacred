#ifndef SACRED_SOFTMAX_GRADIENT_HPP_
#define SACRED_SOFTMAX_GRADIENT_HPP_

#include "default_types.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class SoftmaxGradient : public Operator<F> {
public:
  USING_TENSOR_TYPES(F);

  SoftmaxGradient() = default;

  virtual ~SoftmaxGradient() = default;

  void operator ()(const tensor_type &delta, const tensor_type &y, tensor_type &x_gradient) {
    for (auto i = 0; i < x_gradient.size(); ++i) {
      x_gradient.data(i) = F(0);
      for (auto j = 0; j < delta.size(); ++j) {
        x_gradient.data(i) += y.data(j) * delta.data(j) * ((i == j) - y.data(i));
      }
    }
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto delta = in.at(0), y = in.at(1);
    auto x_gradient = out.at(0);
    operator ()(*delta, *y, *x_gradient);
  }
};

}  // namespace sacred

#endif  // SACRED_SOFTMAX_GRADIENT_HPP_
