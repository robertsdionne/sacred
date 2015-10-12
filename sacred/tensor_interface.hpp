#ifndef SACRED_TENSOR_INTERFACE_HPP_
#define SACRED_TENSOR_INTERFACE_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "checks.hpp"
#include "functional.hpp"
#include "slice.hpp"
#include "testing.hpp"

namespace sacred {

  using namespace std;

  template <typename F> class Tensor;
  // template <typename F> class TensorIterator;
  // template <typename F> class ConstantTensorIterator;

  template <typename F>
  class TensorInterface {
  public:
    ~TensorInterface() = default;

    virtual operator F() const = 0;

    // broadcasting
    virtual Tensor<F> at(const vector<Slice> &indices) = 0;

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

    virtual const int number_of_axes() const = 0;

    // broadcasting and wrapping
    virtual Tensor<F> operator [](const vector<Slice> &indices) = 0;

    virtual Tensor<F> &operator =(F other) = 0;

    virtual Tensor<F> &operator =(const Tensor<F> &other) = 0;

    virtual bool operator ==(const Tensor<F> &other) const = 0;

    virtual const vector<int> &shape() const = 0;

    virtual int size() const = 0;
  };

}  // namespace sacred

#endif  // SACRED_TENSOR_INTERFACE_HPP_
