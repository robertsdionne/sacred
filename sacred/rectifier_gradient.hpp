#ifndef SACRED_RECTIFIER_GRADIENT_HPP_
#define SACRED_RECTIFIER_GRADIENT_HPP_

#include <glog/logging.h>

#include "default_types.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class RectifierGradient : public Operator<F> {
public:
  using tensor_type = typename default_tensor_type<F>::value;
  using tensors_type = typename default_tensors_type<F>::value;
  using tensors_const_type = typename default_tensors_const_type<F>::value;

  RectifierGradient() = default;

  virtual ~RectifierGradient() = default;

  void operator ()(const tensor_type &delta, const tensor_type &x, tensor_type &x_gradient) {
    CHECK_EQ(x.order(), delta.order());
    // TODO(robertsdionne): Support arbitrary tensor order with an n-dimensional iterator.
    for (auto i = 0; i < x.shape().at(0); ++i) {
      for (auto j = 0; j < x.shape().at(1); ++j) {
        x_gradient.set({i, j}, delta.at({i, j}) * (x.at({i, j}) > F(0)));
      }
    }
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto delta = in.at(0), x = in.at(1);
    auto x_gradient = out.at(0);
    operator ()(*delta, *x, *x_gradient);
  }
};

}  // namespace sacred

#endif  // SACRED_RECTIFIER_GRADIENT_HPP_
