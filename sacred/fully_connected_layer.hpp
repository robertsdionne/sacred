#ifndef SACRED_FULLY_CONNECTED_LAYER_HPP_
#define SACRED_FULLY_CONNECTED_LAYER_HPP_

#include "blob.hpp"
#include "layer.hpp"
#include "math.hpp"

namespace sacred {

  template <typename F>
  class FullyConnectedLayer : public Layer<F> {
  public:

    void Forward(const Blob<F> &bottom, Blob<F> &top) override {
    }

    void Backward(const Blob<F> &top, Blob<F> &bottom) override {
    }

  private:
    Blob<F> weights;
    Blob<F> bias;
  };

}  // namespace sacred

#endif  // SACRED_FULLY_CONNECTED_LAYER_HPP_
