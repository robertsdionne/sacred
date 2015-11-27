#ifndef SACRED_RECTIFIED_LINEAR_HPP_
#define SACRED_RECTIFIED_LINEAR_HPP_

#include <algorithm>
#include <glog/logging.h>

#include "default_types.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class RectifiedLinear : public Operator<F> {
public:
  USING_TENSOR_TYPES(F);

  RectifiedLinear() = default;

  virtual ~RectifiedLinear() = default;

  void operator ()(const tensor_type &x, tensor_type &y) {
    using std::max;
    CHECK_EQ(x.order(), y.order());
    // TODO(robertsdionne): Support arbitrary tensor order with an n-dimensional iterator.
    for (auto i = 0; i < x.shape().at(0); ++i) {
      for (auto j = 0; j < x.shape().at(1); ++j) {
        y.set({i, j}, max(F(0), x.at({i, j})));
      }
    }
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto x = in.at(0);
    auto y = out.at(0);
    operator ()(*x, *y);
  }
};

}  // namespace sacred

#endif  // SACRED_RECTIFIED_LINEAR_HPP_
