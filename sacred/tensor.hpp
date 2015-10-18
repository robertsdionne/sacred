#ifndef SACRED_TENSOR_HPP_
#define SACRED_TENSOR_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#include "checks.hpp"
#include "functional.hpp"
#include "slice.hpp"
#include "strides.hpp"
#include "tensor_interface.hpp"
#include "testing.hpp"

namespace sacred {

using namespace std;

template <typename F>
struct TensorEntry {
  vector<int> index;
  F value;
};

template <typename F>
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

  virtual Tensor<F> at(const vector<Slice> &indices) override {
    CHECK_STATE(indices.size() <= shape_.size());
    auto index = 0;
    for (auto i = 0; i < shape_.size(); ++i) {
      CHECK_STATE(0 <= indices.at(i).start(shape_.at(i)) && indices.at(i).start(shape_.at(i)) < shape_.at(i));
      index += indices.at(i).start(shape_.at(i)) * stride_.at(i);
    }
    return data_.at(index);
  }

  virtual const int number_of_axes() const override {
    return shape_.size();
  }

  virtual Tensor<F> operator [](const vector<Slice> &indices) override {
    CHECK_STATE(indices.size() <= shape_.size());
    auto one_dimensional_index = 0;
    for (auto i = 0; i < shape_.size(); ++i) {
      auto index = indices.at(i).start(shape_.at(i)) % shape_.at(i);
      auto wrapped_index = index % shape_.at(i) + shape_.at(i) * (index < 0);
      one_dimensional_index += wrapped_index * stride_.at(i);
    }
    return data_.at(one_dimensional_index);
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
