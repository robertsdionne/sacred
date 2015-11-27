#ifndef SACRED_CONVOLUTION_GRADIENT_HPP_
#define SACRED_CONVOLUTION_GRADIENT_HPP_

#include "default_types.hpp"
#include "operator.hpp"

namespace sacred {

template <typename F = default_floating_point_type>
class ConvolutionGradient : public Operator<F> {
public:
  USING_TENSOR_TYPES(F);

  ConvolutionGradient(tensor_type &bias_gradient, tensor_type &filter, tensor_type &filter_gradient):
      bias_gradient_(bias_gradient), filter_(filter), filter_gradient_(filter_gradient) {}

  virtual ~ConvolutionGradient() = default;

  void operator ()(const tensor_type &delta, const tensor_type &x, tensor_type &x_gradient) {

    for (auto i = 0; i < delta.shape().at(0); ++i) {
      for (auto j = 0; j < delta.shape().at(1); ++j) {
        for (auto k = 0; k < delta.shape().at(2); ++k) {
          bias_gradient_.add({k}, delta.at({i, j, k}));
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
  tensor_type &bias_gradient_, &filter_, &filter_gradient_;
};

}  // namespace sacred

#endif  // SACRED_CONVOLUTION_GRADIENT_HPP_
