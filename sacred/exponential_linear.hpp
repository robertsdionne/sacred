#ifndef SACRED_EXPONENTIAL_LINEAR_HPP_
#define SACRED_EXPONENTIAL_LINEAR_HPP_

#include <algorithm>
#include <glog/logging.h>

#include "default_types.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class ExponentialLinear : public Operator<F> {
public:
  USING_TENSOR_TYPES(F);

  ExponentialLinear(F alpha = F(1)): alpha_(alpha) {}

  virtual ~ExponentialLinear() = default;

  void operator ()(const tensor_type &x, tensor_type &y) {
    using std::exp;
    using std::max;
    CHECK_EQ(x.order(), y.order());
    for (auto i = 0; i < x.size(); ++i) {
      if (F(0) < x.data(i)) {
        y.data(i) = x.data(i);
      } else {
        y.data(i) = alpha_ * (exp(x.data(i)) - F(1));
      }
    }
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto x = in.at(0);
    auto y = out.at(0);
    operator ()(*x, *y);
  }

private:
  F alpha_;
};

}  // namespace sacred

#endif  // SACRED_EXPONENTIAL_LINEAR_HPP_
