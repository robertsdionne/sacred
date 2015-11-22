#ifndef SACRED_FULLY_CONNECTED_HPP_
#define SACRED_FULLY_CONNECTED_HPP_

#include <glog/logging.h>

#include "default_types.hpp"
#include "math.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class FullyConnected : public Operator<F> {
public:
  using tensor_type = typename default_tensor_type<F>::value;
  using tensors_type = typename default_tensors_type<F>::value;
  using tensors_const_type = typename default_tensors_const_type<F>::value;

  FullyConnected(tensor_type &bias, tensor_type &weight) : bias_(bias), weight_(weight) {}

  virtual ~FullyConnected() = default;

  void operator ()(const tensor_type &x, tensor_type &y) {
    CHECK_LE(x.order(), 2);
    CHECK_GT(x.order(), 0);
    CHECK_LE(y.order(), 2);
    CHECK_GT(y.order(), 0);
    CHECK_EQ(x.order(), y.order());
    math_.GeneralMatrixMultiplication(y, weight_, x, F(0), F(1));
    math_.Add(y, bias_, F(1), F(1));
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto x = in.at(0);
    auto y = out.at(0);
    operator ()(*x, *y);
  }

private:
  tensor_type &bias_, &weight_;
  Math<F> math_;
};

}  // namespace sacred

#endif  // SACRED_FULLY_CONNECTED_HPP_
