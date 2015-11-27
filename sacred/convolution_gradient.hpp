#ifndef SACRED_CONVOLUTION_GRADIENT_HPP_
#define SACRED_CONVOLUTION_GRADIENT_HPP_

#include <algorithm>

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
    using std::min;
    using std::max;

    // input gradient
    for (auto i = 0; i < x_gradient.shape().at(0); ++i) {
      for (auto j = 0; j < x_gradient.shape().at(1); ++j) {

        auto s_begin = max(0, i - (x_gradient.shape().at(0) - filter_.shape().at(0))),
            s_end = min(filter_.shape().at(0), i + 1);

        auto t_begin = max(0, j - (x_gradient.shape().at(1) - filter_.shape().at(1))),
            t_end = min(filter_.shape().at(1), j + 1);

        for (auto s = s_begin; s < s_end; ++s) {
          for (auto t = t_begin; t < t_end; ++t) {
            for (auto k = 0; k < x_gradient.shape().at(2); ++k) {
              F output_value = F(1) * x_gradient.at({i, j, k});
              for (auto u = 0; u < filter_.shape().at(2); ++u) {
                output_value += F(1) * filter_.at({s, t, u, k}) * delta.at({i - s, j - t, u});
              }
              x_gradient.set({i, j, k}, output_value);
            }
          }
        }
      }
    }

    // bias gradient
    for (auto i = 0; i < delta.shape().at(0); ++i) {
      for (auto j = 0; j < delta.shape().at(1); ++j) {
        for (auto k = 0; k < delta.shape().at(2); ++k) {
          bias_gradient_.add({k}, delta.at({i, j, k}));
        }
      }
    }

    // filter gradient
    for (auto s = 0; s < filter_gradient_.shape().at(0); ++s) {
      for (auto t = 0; t < filter_gradient_.shape().at(1); ++t) {
        for (auto u = 0; u < filter_gradient_.shape().at(2); ++u) {
          for (auto v = 0; v < filter_gradient_.shape().at(3); ++v) {
            F output_value = F(1) * filter_gradient_.at({s, t, u, v});
            for (auto i = 0; i < delta.shape().at(0); ++i) {
              for (auto j = 0; j < delta.shape().at(1); ++j) {
                output_value += F(1) * delta.at({i, j, u}) * x.at({i + s, j + t, v});
              }
            }
            filter_gradient_.set({s, t, u, v}, output_value);
          }
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
