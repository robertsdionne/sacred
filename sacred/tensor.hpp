#ifndef SACRED_TENSOR_HPP_
#define SACRED_TENSOR_HPP_

#include <initializer_list>
#include <vector>

#include "checks.hpp"

namespace sacred {

  using std::initializer_list;
  using std::vector;

  template <typename F>
  class Tensor {
  public:
    Tensor(initializer_list<int> shape, const vector<F> &data) : shape_(shape), data_(data) {}

    ~Tensor() = default;

    F at(int index) {
      return data_.at(index);
    }

    F at(initializer_list<int> indices) {
      CHECK_STATE(indices.size() == shape_.size());
      int index = 0;
      initializer_list<int>::iterator s, i;
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

    inline int shape(int index) {
      CHECK_STATE(0 <= index && index < shape_.size());
      return *(shape_.begin() + index);
    }

    inline int size() const {
      return data_.size();
    }

  private:
    initializer_list<int> shape_;
    vector<F> data_;
  };

}  // namespace sacred

#endif  // SACRED_TENSOR_HPP_
