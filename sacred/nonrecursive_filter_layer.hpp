#ifndef SACRED_NONRECURSIVE_FILTER_LAYER_HPP_
#define SACRED_NONRECURSIVE_FILTER_LAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"
#include "math.hpp"

namespace sacred {

  template <typename F>
  class NonrecursiveFilterLayer : public Layer<F> {
  public:
    NonrecursiveFilterLayer(const Blob<F> &bias, const Blob<F> filter) : bias_(bias), filter_(filter) {}

    virtual ~NonrecursiveFilterLayer() = default;

    void Forward(const Blob<F> &bottom, Blob<F> *top) override {
      // y = c ⁎ u + b
      // y_n = Σ_k c_k * u_{n-k} + b_n
      math_.NarrowConvolve2(top->value(), filter_.value(), bottom.value(), F(1.0), F(1.0));
      math_.Add(top->value(), bias_.value(), F(1.0), F(1.0));
    }

    void Backward(const Blob<F> &top, Blob<F> *bottom) override {
      // Filter derivatives:
      // ∂E/∂c_k = Σ_n ∂E/∂y_n * u_{n-k}
      // ∇_c E = ∇_y E ⁎ u
      math_.NarrowConvolve2(filter_.diff(), top.diff(), bottom->value(), F(1.0), F(1.0));

      // Bottom derivatives:
      // ∂E/∂u_n = Σ_k ∂E/∂y_{n+k} c_k
      // ∇_u E = ∇_y E ⁎ c
      math_.BackwardWideConvolve2(bottom->diff(), filter_.value(), top.diff(), F(1.0), F(1.0));

      // Bias derivatives:
      // ∂E/∂b_n = ∂E/∂y_n
      // ∇_b E = ∇_y E
      math_.Add(bias_.diff(), top.diff(), F(1.0), F(1.0));
    }

  private:
    Blob<F> bias_, filter_;
    Math<F> math_;
  };

}  // namespace sacred

#endif  // SACRED_NONRECURSIVE_FILTER_LAYER_HPP_
