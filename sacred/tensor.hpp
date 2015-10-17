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
    Tensor() : shape_({1}), data_({F()}) {}

    Tensor(F value) : shape_({1}), data_({value}) {}

    Tensor(const vector<int> &shape) : shape_(shape), data_(ProductOf(shape)) {}

    Tensor(const vector<int> &shape, const vector<F> &data) : shape_(shape), data_(data) {}

    ~Tensor() = default;

    virtual operator F() const override {
      CHECK_STATE(1 == ProductOf(shape_));
      return data_.at(0);
    }

    virtual Tensor<F> at(const vector<Slice> &indices) override {
      // auto shape = SelectFrom(shape_, slices);
      // if (1 == shape.size()) {
      //   return SliceBase(indices, slices.at(0));
      // }
      return data_.at(0);
    }

    // Tensor<F> SliceBase(const vector<int> &indices, int slice) {
    //   auto shape = shape_.at(slice);
    //   auto subtensor = Tensor<F>({shape});
    //   auto index = vector<int>(indices);
    //   index.insert(next(begin(index), slice), 0);
    //   for (auto i = 0; i < shape; ++i) {
    //     index.at(slice) = i;
    //     subtensor.at(i) = at(index);
    //   }
    //   return subtensor;
    // }

    // F &at(const vector<int> &indices) {
    //   CHECK_STATE(indices.size() == shape_.size());
    //   int index = 0;
    //   vector<int>::const_iterator s, i;
    //   for (s = shape_.begin(), i = indices.begin();
    //       s < shape_.end() && i < indices.end();
    //       ++s, ++i) {
    //     index *= *s;
    //     if (0 <= *i && *i < *s) {
    //       index += *i;
    //     }
    //   }
    //   return data_.at(index);
    // }

    virtual const int number_of_axes() const override {
      return shape_.size();
    }

    virtual Tensor<F> operator [](const vector<Slice> &indices) override {
      return data_.at(0);
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
    vector<int> shape_;
    vector<F> data_;
  };

}  // namespace sacred

#endif  // SACRED_TENSOR_HPP_
