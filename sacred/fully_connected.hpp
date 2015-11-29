#ifndef SACRED_FULLY_CONNECTED_HPP_
#define SACRED_FULLY_CONNECTED_HPP_

#include <glog/logging.h>

#include "default_types.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class FullyConnected : public Operator<F> {
public:
  USING_TENSOR_TYPES(F);

  FullyConnected(tensor_type &bias, tensor_type &weight) : bias_(bias), weight_(weight) {}

  virtual ~FullyConnected() = default;

  void operator ()(const tensor_type &x, tensor_type &y) {
    for (auto i = 0; i < y.shape().at(0); ++i) {
      for (auto q = 0; q < y.shape().at(1); ++q) {
        for (auto r = 0; r < y.shape().at(2); ++r) {
          F current_output = F(1) * y.at({i, q, r});
          for (auto j = 0; j < x.shape().at(1); ++j) {
            for (auto k = 0; k < x.shape().at(2); ++k) {
              current_output += F(1) * weight_.at({q, r, j, k}) * x.at({i, j, k});
            }
          }
          y.set({i, q, r}, F(1) * current_output + F(1) * bias_.at({q, r}));
        }
      }
    }
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto x = in.at(0);
    auto y = out.at(0);
    operator ()(*x, *y);
  }

private:
  tensor_type &bias_, &weight_;
};

}  // namespace sacred

#endif  // SACRED_FULLY_CONNECTED_HPP_
