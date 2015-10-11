#ifndef SACRED_TENSOR_HPP_
#define SACRED_TENSOR_HPP_

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

  struct Slice {};

  Slice _;

  template <typename F>
  class Tensor {
  public:
    Tensor(const vector<int> &shape) : shape_(shape), data_(ProductOf(shape)) {}

    Tensor(const vector<int> &shape, const vector<F> &data) : shape_(shape), data_(data) {}

    ~Tensor() = default;

    template <typename... Parameter>
    Tensor<F> slice(int first, Parameter... rest) {
      vector<int> indices({first}), slices;
      return slice(indices, slices, rest...);
    }

    template <typename... Parameter>
    Tensor<F> slice(Slice first, Parameter... rest) {
      vector<int> indices, slices({0});
      return slice(indices, slices, rest...);
    }

    template <typename... Parameter>
    Tensor<F> slice(vector<int> &indices, vector<int> &slices, int next, Parameter... rest) {
      indices.push_back(next);
      return slice(indices, slices, rest...);
    }

    template <typename... Parameter>
    Tensor<F> slice(vector<int> &indices, vector<int> &slices, Slice next, Parameter... rest) {
      slices.push_back(indices.size() + slices.size());
      return slice(indices, slices, rest...);
    }

    Tensor<F> slice(const vector<int> &indices, const vector<int> &slices) {
      auto shape = SelectFrom(shape_, slices);
      if (1 == shape.size()) {
        return SliceBase(indices, slices.at(0));
      }
      auto subtensor = Tensor<F>(shape);
      return subtensor;
    }

    Tensor<F> SliceBase(const vector<int> &indices, int slice) {
      auto shape = shape_.at(slice);
      auto subtensor = Tensor<F>({shape});
      auto index = vector<int>(indices);
      index.insert(next(begin(index), slice), 0);
      for (auto i = 0; i < shape; ++i) {
        index.at(slice) = i;
        subtensor.at(i) = at(index);
      }
      return subtensor;
    }

    template <typename... Int>
    F &at(int index, Int... rest) {
      vector<int> indices({index});
      return at(indices, rest...);
    }

    template <typename... Int>
    F &at(vector<int> &indices, int next, Int... rest) {
      indices.push_back(next);
      return at(indices, rest...);
    }

    F &at(const vector<int> &indices) {
      CHECK_STATE(indices.size() == shape_.size());
      int index = 0;
      vector<int>::const_iterator s, i;
      for (s = shape_.begin(), i = indices.begin();
          s < shape_.end() && i < indices.end();
          ++s, ++i) {
        index *= *s;
        if (0 <= *i && *i < *s) {
          index += *i;
        }
      }
      return data_.at(index);
    }

    const int number_of_axes() const {
      return shape_.size();
    }

    friend bool operator ==(const Tensor<F> &left, const Tensor<F> &right) {
      return left.shape_ == right.shape_ && left.data_ == right.data_;
    }

    friend ostream &operator <<(ostream &out, const Tensor<F> &tensor) {
      return out << "Tensor<F>(" << tensor.shape_ << ", " << tensor.data_ << ")";
    }

    inline int shape(int index) {
      CHECK_STATE(0 <= index && index < shape_.size());
      return *(shape_.begin() + index);
    }

    inline int size() const {
      return data_.size();
    }

  private:
    vector<int> shape_;
    vector<F> data_;
  };

}  // namespace sacred

#endif  // SACRED_TENSOR_HPP_
