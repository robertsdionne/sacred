#ifndef SACRED_RECURSIVE_FILTER_LAYER_HPP_
#define SACRED_RECURSIVE_FILTER_LAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"
#include "math.hpp"

namespace sacred {

template <typename F>
class RecursiveFilterLayer : public Layer<F> {
public:
  RecursiveFilterLayer(Blob<F> &bias, Blob<F> &filter) : bias_(bias), filter_(filter) {}

  virtual ~RecursiveFilterLayer() = default;

  void Forward(const Blob<F> &bottom, Blob<F> *top) override {
    // y = c ⁎ u + b
    // y_n = Σ_k c_k * u_{n-k} + b_n
    math_.Add(top->value(), bottom.value(), F(1.0), F(1.0));
    math_.RecurrentConvolve2(top->value(), filter_.value(), F(1.0), F(1.0));
    math_.BroadcastAdd(top->value(), bias_.value(), F(1.0), F(1.0));
  }

  void Backward(const Blob<F> &top, Blob<F> *bottom) override {
    // Filter derivatives:
    // ∂E/∂c_k = Σ_n ∂E/∂y_n * u_{n+k}
    // Note: my manual calculations figured ∂E/∂c_k = Σ_n ∂E/∂y_n * u_{n-k} but this disagrees
    //    with the dual number gradient verification. My mistake might be that
    //    ∂E/∂c_{-k} = Σ_n ∂E/∂y_n * u_{n-k} === ∂E/∂c_k = Σ_n ∂E/∂y_n * u_{n+k} is correct.
    // ∇_c E = ∇_y E ⁎ u
    math_.BackwardRecurrentConvolveFilter(filter_.diff(), filter_.value(), top.diff(), top.value());

    // Bottom derivatives:
    // ∂E/∂u_n = Σ_k ∂E/∂y_{n+k} c_k
    // ∇_u E = ∇_y E ⁎ c
    math_.Add(bottom->diff(), top.diff(), F(1.0), F(1.0));
    math_.BackwardRecurrentConvolve2(bottom->diff(), filter_.value(), F(1.0), F(1.0));

    // Bias derivatives:
    // ∂E/∂b_n = ∂E/∂y_n
    // ∇_b E = ∇_y E
    math_.Sum(bias_.diff(), top.diff(), F(1.0), F(1.0));
  }

private:
  Blob<F> &bias_, &filter_;
  Math<F> math_;
};

}  // namespace sacred

#endif  // SACRED_RECURSIVE_FILTER_LAYER_HPP_
