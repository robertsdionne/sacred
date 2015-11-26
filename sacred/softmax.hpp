#ifndef SACRED_SOFTMAX_HPP_
#define SACRED_SOFTMAX_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

#include "default_types.hpp"
#include "operator.hpp"

namespace sacred {

using std::exp;
using std::max_element;
using std::numeric_limits;

template <typename F = default_floating_point_type>
class Softmax : public Operator<F> {
public:
  USING_TENSOR_TYPES();

  Softmax() = default;

  virtual ~Softmax() = default;

  void operator ()(const tensor_type &x, tensor_type &y) {
    auto maximum = max_element(x.data().begin(), x.data().end());
    auto sum = F(1.19209e-07);
    for (auto i = 0; i < x.size(); ++i) {
      sum += exp(x.data(i) - *maximum);
    }
    for (auto i = 0; i < x.size(); ++i) {
      y.data(i) = exp(x.data(i) - *maximum + F(1.19209e-07)) / sum;
    }
  }

  virtual void operator ()(const tensors_const_type &in, const tensors_type &out) override {
    auto x = in.at(0);
    auto y = out.at(0);
    operator ()(*x, *y);
  }
};

}  // sacred

#endif  // SACRED_SOFTMAX_HPP_
