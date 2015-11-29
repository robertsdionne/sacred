#ifndef SACRED_TENSOR_HPP_
#define SACRED_TENSOR_HPP_

#include <algorithm>
#include <glog/logging.h>
#include <iostream>
#include <iterator>
#include <ostream>
#include <type_traits>

#include "default_types.hpp"
#include "functional.hpp"
#include "indexing/checked_index.hpp"
#include "indexing/identity_lookup.hpp"
#include "indexing/index_strategy.hpp"
#include "indexing/lookup_strategy.hpp"
#include "indexing/strides.hpp"
#include "indexing/wrapped_index.hpp"
#include "tensor_interface.hpp"
#include "testing.hpp"

namespace sacred {

using namespace std;

template <typename F = default_floating_point_type, typename I = default_integer_type>
struct TensorEntry {
  using index_type = typename default_index_type<I>::value;

  index_type index;
  F value;
};

template <typename F = default_floating_point_type, typename I = default_integer_type>
class Tensor : public TensorInterface<F, I> {
public:
  using storage_type = typename default_storage_type<F>::value;
  using index_type = typename default_index_type<I>::value;
  using tensor_type = Tensor<F, I>;

  Tensor(): shape_({1}), stride_(shape_.size()), data_({F()}) {}

  Tensor(F value): shape_({1}), stride_(shape_.size()), data_({value}) {}

  Tensor(const index_type &shape) : shape_(shape), stride_(indexing::strides::CStyle(shape)), data_(ProductOf(shape)) {}

  Tensor(const index_type &shape, const storage_type &data):
      shape_(shape), stride_(indexing::strides::CStyle(shape)), data_(data) {}

  virtual ~Tensor() = default;

  virtual operator F() const override {
    CHECK(1 == ProductOf(shape_));
    return data_.at(0);
  }

  // IndexStrategy
  // * CheckedIndex
  //   * Checks indices lie within shape
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
  // * MaskedLookup
  //   * Looks up values within shape
  //   * Returns default value without
  // * HashedLookup
  //   * Looks up values with a hashed strategy
  //
  // at(): {CheckedIndex} x {IdentityLookup}
  // operator[]: {WrappedIndex} x {IdentityLookup}
  // others: {IdentityIndex} x {IdentityLookup, MaskedLookup, HashedLookup}
  //         {CheckedIndex, WrappedIndex, ClippedIndex, MirroredIndex} x {IdentityLookup, HashedLookup}
  //
  // Note: Use std::valarray, std::slice, std::gslice.
  template <typename Index = indexing::CheckedIndex<I>, typename Lookup = indexing::IdentityLookup<F, I>>
  F &at(const index_type &index) {
    static_assert(is_base_of<indexing::IndexStrategy<I>, Index>::value,
        "Index must implement interface IndexStrategy<I>.");
    static_assert(is_base_of<indexing::LookupStrategy<F, I>, Lookup>::value,
        "Lookup must implement interface LookupStrategy<F, I>.");
    auto transformed_index = Index().Transform(shape_, stride_, index);
    return Lookup().Lookup(data_, data_.size(), shape_, stride_, transformed_index);
  }

  template <typename Index = indexing::CheckedIndex<I>, typename Lookup = indexing::IdentityLookup<F, I>>
  F at(const index_type &index) const {
    static_assert(is_base_of<indexing::IndexStrategy<I>, Index>::value,
        "Index must implement interface IndexStrategy<I>.");
    static_assert(is_base_of<indexing::LookupStrategy<F, I>, Lookup>::value,
        "Lookup must implement interface LookupStrategy<F, I>.");
    auto transformed_index = Index().Transform(shape_, stride_, index);
    return Lookup().Lookup(data_, data_.size(), shape_, stride_, transformed_index);
  }

  storage_type &data() {
    return data_;
  }

  const storage_type &data() const {
    return data_;
  }

  F &data(const I index) {
    return data_.at(index);
  }

  F data(const I index) const {
    return data_.at(index);
  }

  index_type index(I index) const {
    auto result = index_type(stride_.size());
    for (auto i = 0; i < stride_.size(); ++i) {
      result.at(i) = index / stride_.at(i);
      index %= stride_.at(i);
    }
    return result;
  }

  template <typename Index = indexing::CheckedIndex<I>, typename Lookup = indexing::IdentityLookup<F, I>>
  F add(const index_type &index, const F x) {
    return at<Index, Lookup>(index) += x;
  }

  template <typename Index = indexing::CheckedIndex<I>, typename Lookup = indexing::IdentityLookup<F, I>>
  F set(const index_type &index, const F x) {
    return at<Index, Lookup>(index) = x;
  }

  virtual const I number_of_axes() const override {
    return shape_.size();
  }

  virtual tensor_type operator [](const index_type &index) override {
    return at<indexing::WrappedIndex<I>, indexing::IdentityLookup<F, I>>(index);
  }

  virtual void operator =(F other) override {
    for (auto &entry : data_) {
      entry = other;
    }
  }

  virtual void operator =(const tensor_type &other) override {
    // for (auto entry : other) {
    //   at(entry.index) = entry.value;
    // }
  }

  virtual bool operator ==(const tensor_type &other) const override {
    return shape_ == other.shape_ && data_ == other.data_;
  }

  friend ostream &operator <<(ostream &out, const tensor_type &tensor) {
    return out << "Tensor<F>(" << tensor.shape_ << ", " << tensor.data_ << ")";
  }

  virtual inline I order() const override {
    return shape_.size();
  }

  virtual inline const index_type &shape() const override {
    return shape_;
  }

  virtual inline I size() const override {
    return data_.size();
  }

private:
  index_type shape_, stride_;
  storage_type data_;
};

}  // namespace sacred

#endif  // SACRED_TENSOR_HPP_
