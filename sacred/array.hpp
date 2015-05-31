#ifndef SACRED_ARRAY_HPP_
#define SACRED_ARRAY_HPP_

#include <limits>
#include <ostream>
#include <vector>

#include "checks.hpp"

namespace sacred {

  using std::numeric_limits;
  using std::ostream;
  using std::vector;

  template <typename F>
  class Array {
  public:
    Array() = default;

    explicit Array(const vector<int> &shape) : capacity_(), count_(), shape_(shape) {
      Reshape(shape);
    }

    Array(const vector<int> &shape, const vector<F> &data)
        : capacity_(data.size()), count_(), shape_(shape), data_(data) {
      Reshape(shape);
    }

    ~Array() = default;

    const F &data(int index) const {
      return data_.at(index);
    }

    F &data(int index) {
      return data_.at(index);
    }

    F at(const vector<int> &indices) const {
      auto offset = Offset(indices);
      if (offset < 0) {
        return F(0.0);
      }
      return data_.at(offset);
    }

    F add(const vector<int> &indices, const F x) {
      auto offset = Offset(indices);
      if (offset < 0) {
        return F(0.0);
      }
      data_.at(offset) += x;
      return data_.at(offset);
    }

    F axpby(const vector<int> &indices, const F alpha, const F x, const F beta) {
      auto offset = Offset(indices);
      if (offset < 0) {
        return F(0.0);
      }
      data_.at(offset) *= beta;
      data_.at(offset) += alpha * x;
      return data_.at(offset);
    }

    F set(const vector<int> &indices, const F x) {
      auto offset = Offset(indices);
      if (offset < 0) {
        return F(0.0);
      }
      data_.at(offset) = x;
      return data_.at(offset);
    }

    inline int count() const {
      return count_;
    }

    inline const F *data() const {
      return data_.data();
    }

    inline F *data() {
      return data_.data();
    }

    inline int number_of_axes() const {
      return shape_.size();
    }

    bool operator ==(const Array<F> &other) const {
      if (number_of_axes() != other.number_of_axes()) {
        return false;
      }
      for (auto i = 0; i < number_of_axes(); ++i) {
        if (shape(i) != other.shape(i)) {
          return false;
        }
      }
      for (auto i = 0; i < count(); ++i) {
        if (data(i) != other.data(i)) {
          return false;
        }
      }
      return true;
    }

    inline const vector<int> &shape() const {
      return shape_;
    }

    inline int shape(int index) const {
      return shape_.at(index);
    }

    void Reshape(const vector<int> &shape) {
      count_ = 1;
      shape_.resize(shape.size());
      for (auto i = 0; i < shape.size(); ++i) {
        CHECK_STATE(0 <= shape.at(i));
        CHECK_STATE(shape.at(i) < numeric_limits<int>::max() / count_);
        count_ *= shape.at(i);
        shape_.at(i) = shape.at(i);
      }
      if (count_ > capacity_) {
        capacity_ = count_;
        data_.resize(capacity_);
      }
    }

    int Offset(const vector<int> &indices) const {
      CHECK_STATE(indices.size() == number_of_axes());
      int offset = 0;
      for (auto i = 0; i < number_of_axes(); ++i) {
        offset *= shape(i);
        if (indices.size() > i) {
          CHECK_STATE(0 <= indices.at(i));
          CHECK_STATE(indices.at(i) < shape(i));
          offset += indices.at(i);
        }
      }
      return offset;
    }

  private:
    int capacity_, count_;
    vector<int> shape_;
    vector<F> data_;
  };

  template <typename F>
  ostream &operator <<(ostream &out, const Array<F> &array) {
    for (auto i = 0; i < array.shape(0); ++i) {
      for (auto j = 0; j < array.shape(1); ++j) {
        out << array.at({i, j}) << u8" ";
      }
      out << std::endl;
    }
    return out;
  }

}  // namespace sacred

#endif  // SACRED_ARRAY_HPP_
