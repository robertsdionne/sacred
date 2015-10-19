#ifndef SACRED_TENSOR_HPP_
#define SACRED_TENSOR_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>
#include <type_traits>
#include <vector>

#include "checked_lookup.hpp"
#include "checks.hpp"
#include "functional.hpp"
#include "identity_index.hpp"
#include "index_strategy.hpp"
#include "lookup_strategy.hpp"
#include "slice.hpp"
#include "strides.hpp"
#include "tensor_interface.hpp"
#include "testing.hpp"
#include "wrapped_index.hpp"

namespace sacred {

using namespace std;

template <typename F>
struct TensorEntry {
  vector<int> index;
  F value;
};

template <typename F = float>
class Tensor : public TensorInterface<F> {
public:
  Tensor(): shape_({1}), stride_(shape_.size()), data_({F()}) {}

  Tensor(F value): shape_({1}), stride_(shape_.size()), data_({value}) {}

  Tensor(const vector<int> &shape) : shape_(shape), stride_(strides::CStyle(shape)), data_(ProductOf(shape)) {}

  Tensor(const vector<int> &shape, const vector<F> &data):
      shape_(shape), stride_(strides::CStyle(shape)), data_(data) {}

  ~Tensor() = default;

  virtual operator F() const override {
    CHECK_STATE(1 == ProductOf(shape_));
    return data_.at(0);
  }

  // IndexStrategy
  // * IdentityIndex
  //   * Passes indices through unchanged
  // * WrappedIndex
  //   * Wraps indices about shape
  //   * Implies indices lie within shape
  // * ClippedIndex
  //   * Clips indices to range
  //   * Implies indices lie within shape
  // * MirroredIndex
  //   * Mirrors the index about shape
  //   * Implies indices lie within shape
  // LookupStrategy
  // * IdentityLookup
  //   * Looks up values directly
  // * CheckedLookup
  //   * Checks indices lie within shape
  // * MaskedLookup
  //   * Looks up values within shape
  //   * Returns default value without
  // * HashedLookup
  //   * Looks up values with a hashed strategy
  //
  // at(): {IdentityIndex} x {CheckedLookup}
  // operator[]: {IdentityIndex} x {IdentityLookup, MaskedLookup, HashedLookup}
  //             {WrappedIndex, ClippedIndex, MirroredIndex} x {IdentityLookup, HashedLookup}
  template <typename Index = IdentityIndex, typename Lookup = CheckedLookup>
  Tensor<F> at(const vector<int> &index) {
    static_assert(is_base_of<tensor::IndexStrategy, Index>::value, "Index must implement interface IndexStrategy.");
    static_assert(is_base_of<tensor::LookupStrategy, Lookup>::value, "Lookup must implement interface LookupStrategy.");
    return data_.at(Lookup().Offset(data_.size(), shape_, stride_, Index().Transform(shape_, stride_, index)));
  }

  virtual const int number_of_axes() const override {
    return shape_.size();
  }

  virtual Tensor<F> operator [](const vector<int> &index) override {
    return at<WrappedIndex, IdentityLookup>(index);
  }

  virtual Tensor<F> &operator =(F other) override {
    for (auto &entry : data_) {
      entry = other;
    }
    return *this;
  }

  virtual Tensor<F> &operator =(const Tensor<F> &other) override {
    // for (auto entry : other) {
    //   at(entry.index) = entry.value;
    // }
    return *this;
  }

  virtual bool operator ==(const Tensor<F> &other) const override {
    return shape_ == other.shape_ && data_ == other.data_;
  }

  friend ostream &operator <<(ostream &out, const Tensor<F> &tensor) {
    return out << "Tensor<F>(" << tensor.shape_ << ", " << tensor.data_ << ")";
  }

  virtual inline const vector<int> &shape() const override {
    return shape_;
  }

  virtual inline int size() const override {
    return data_.size();
  }

private:
  vector<int> shape_, stride_;
  vector<F> data_;
};

}  // namespace sacred

#endif  // SACRED_TENSOR_HPP_
