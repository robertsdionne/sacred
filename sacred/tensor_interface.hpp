#ifndef SACRED_TENSOR_INTERFACE_HPP_
#define SACRED_TENSOR_INTERFACE_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "checks.hpp"
#include "functional.hpp"
#include "testing.hpp"

namespace sacred {

using namespace std;

template <typename F, typename I> class Tensor;
// template <typename F> class TensorIterator;
// template <typename F> class ConstantTensorIterator;

template <typename F, typename I>
class TensorInterface {
public:
  using storage_type = vector<F>;
  using index_type = vector<I>;

  ~TensorInterface() = default;

  virtual operator F() const = 0;

  // broadcasting
  // virtual Tensor<F> at(const vector<int> &indices) = 0;

  // virtual TensorIterator<F> begin() = 0;
  //
  // virtual ConstantTensorIterator<F> begin() const = 0;
  //
  // virtual ConstantTensorIterator<F> cbegin() const = 0;
  //
  // virtual TensorIterator<F> end() = 0;
  //
  // virtual ConstantTensorIterator<F> end() const = 0;
  //
  // virtual ConstantTensorIterator<F> cend() const = 0;

  virtual const I number_of_axes() const = 0;

  // broadcasting and wrapping
  virtual Tensor<F, I> operator [](const index_type &index) = 0;

  virtual Tensor<F, I> &operator =(F other) = 0;

  virtual Tensor<F, I> &operator =(const Tensor<F, I> &other) = 0;

  virtual bool operator ==(const Tensor<F, I> &other) const = 0;

  virtual const index_type &shape() const = 0;

  virtual I size() const = 0;
};

}  // namespace sacred

#endif  // SACRED_TENSOR_INTERFACE_HPP_
