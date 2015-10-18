#ifndef SACRED_LAYER_HPP_
#define SACRED_LAYER_HPP_

#include "blob.hpp"

namespace sacred {

template <typename F>
class Layer {
public:
  virtual ~Layer() = default;

  virtual void Forward(const Blob<F> &bottom, Blob<F> *top) = 0;

  virtual void Backward(const Blob<F> &top, Blob<F> *bottom) = 0;
};

}  // namespace sacred

#endif  // SACRED_LAYER_HPP_
