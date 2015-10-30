#ifndef SACRED_TENSOR_TRAIN_HPP_
#define SACRED_TENSOR_TRAIN_HPP_

#include "default_types.hpp"
#include "functional.hpp"
#include "indexing/checked_index.hpp"
#include "indexing/identity_lookup.hpp"
#include "indexing/index_strategy.hpp"
#include "indexing/lookup_strategy.hpp"
#include "indexing/wrapped_index.hpp"
#include "tensor_interface.hpp"
#include "tensor.hpp"

namespace sacred {

template <typename F = default_floating_point_type, typename I = default_integer_type>
class TensorTrain : public TensorInterface<F, I> {
public:
  using storage_type = typename default_storage_type<F>::value;
  using index_type = typename default_index_type<I>::value;
  using tensor_type = Tensor<F, I>;

  TensorTrain(): shape_({3, 3, 3}), tensors_({
    {{1, 3, 1}, {1, 2, 3}},
    {{1, 3, 1}, {1, 2, 3}},
    {{1, 3, 1}, {1, 2, 3}},
  }) {}

  virtual ~TensorTrain() = default;

  template <typename Index = indexing::CheckedIndex<I>>
  tensor_type at(const index_type &index) {
    static_assert(is_base_of<indexing::IndexStrategy<I>, Index>::value,
        "Index must implement interface IndexStrategy<I>.");
    auto values = storage_type();
    for (auto i = 0; i < index.size(); ++i) {
      values.push_back(tensors_.at(i).at({0, index.at(i), 0}));
    }
    return ProductOf<F>(values);
  }

  virtual operator F() const override {
    CHECK(1 == ProductOf(shape_));
    return F(0);
  }

  virtual const I number_of_axes() const override {
    return shape_.size();
  }

  virtual tensor_type operator [](const index_type &index) override {
    return at<indexing::WrappedIndex<I>>(index);
  }

  virtual void operator =(F other) override {
  }

  virtual void operator =(const tensor_type &other) override {
  }

  virtual bool operator ==(const tensor_type &other) const override {
    return false;
  }

  virtual const index_type &shape() const override {
    return shape_;
  }

  virtual I size() const override {
    return 0;
  }

private:
  index_type shape_;
  vector<tensor_type> tensors_;
};

}  // namespace sacred

#endif  // SACRED_TENSOR_TRAIN_HPP_
