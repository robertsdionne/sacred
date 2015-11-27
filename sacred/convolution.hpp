#ifndef SACRED_CONVOLUTION_HPP_
#define SACRED_CONVOLUTION_HPP_

#include <iostream>

#include "default_types.hpp"
#include "math.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class Convolution : public Operator<F> {
public:
  USING_TENSOR_TYPES(F);

  Convolution(tensor_type &bias, tensor_type &filter, index_type stride={1, 1}):
      bias_(bias), filter_(filter), stride_(stride) {}

  virtual ~Convolution() = default;

  void operator ()(const tensor_type &x, tensor_type &y) {
    // y = F ∗ x + b
    for (auto i = 0; i < y.shape().at(0); i += stride_.at(0)) {
      for (auto j = 0; j < y.shape().at(1); j += stride_.at(1)) {
        for (auto s = 0; s < filter_.shape().at(0); ++s) {
          for (auto t = 0; t < filter_.shape().at(1); ++t) {
            for (auto u = 0; u < y.shape().at(2); ++u) {
              F output_value = F(1) * y.at({i, j, u});
              for (auto v = 0; v < filter_.shape().at(3); ++v) {
                output_value += F(1) * filter_.at({s, t, u, v}) * x.at({i + s, j + t, v});
              }
              y.set({i, j, u}, output_value);
            }
          }
        }
      }
    }

    for (auto i = 0; i < y.shape().at(0); i += stride_.at(0)) {
      for (auto j = 0; j < y.shape().at(1); j += stride_.at(1)) {
        for (auto k = 0; k < y.shape().at(2); ++k) {
          y.set({i, j, k}, F(1) * y.at({i, j, k}) + F(1) * bias_.at({k}));
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
  tensor_type &bias_, &filter_;
  index_type stride_;
  Math<F> math_;
};

}  // namespace sacred

#endif  // SACRED_CONVOLUTION_HPP_
