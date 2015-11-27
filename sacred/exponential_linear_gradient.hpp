#ifndef SACRED_EXPONENTIAL_LINEAR_GRADIENT_HPP_
#define SACRED_EXPONENTIAL_LINEAR_GRADIENT_HPP_

#include <glog/logging.h>

#include "default_types.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class ExponentialLinearGradient : public Operator<F> {
public:
  USING_TENSOR_TYPES(F);

  ExponentialLinearGradient(F alpha = F(1)): alpha_(alpha) {}

  virtual ~ExponentialLinearGradient() = default;

  void operator ()(const tensor_type &y, const tensor_type &delta, const tensor_type &x, tensor_type &x_gradient) {
    CHECK_EQ(x.order(), delta.order());
    for (auto i = 0; i < x.size(); ++i) {
      x_gradient.data(i) = delta.data(i) * (x.data(i) > F(0)) + (y.data(i) + alpha_) * (x.data(i) <= F(0));
    }
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto y = in.at(0), delta = in.at(1), x = in.at(2);
    auto x_gradient = out.at(0);
    operator ()(*y, *delta, *x, *x_gradient);
  }

private:
  F alpha_;
};

}  // namespace sacred

#endif  // SACRED_EXPONENTIAL_LINEAR_GRADIENT_HPP_
