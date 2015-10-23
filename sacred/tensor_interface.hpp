#ifndef SACRED_TENSOR_INTERFACE_HPP_
#define SACRED_TENSOR_INTERFACE_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>

#include "checks.hpp"
#include "default_types.hpp"
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
  using storage_type = typename default_storage_type<F>::value;
  using index_type = typename default_index_type<I>::value;
  using tensor_type = Tensor<F, I>;

  ~TensorInterface() = default;

  virtual operator F() const = 0;

  // broadcasting
  // virtual Tensor<F> at(const index_type &index) = 0;

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
  virtual tensor_type operator [](const index_type &index) = 0;

  virtual tensor_type &operator =(F other) = 0;

  virtual tensor_type &operator =(const tensor_type &other) = 0;

  virtual bool operator ==(const tensor_type &other) const = 0;

  virtual const index_type &shape() const = 0;

  virtual I size() const = 0;
};

}  // namespace sacred

#endif  // SACRED_TENSOR_INTERFACE_HPP_
