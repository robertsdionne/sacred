#ifndef SACRED_ARRAY_HPP_
#define SACRED_ARRAY_HPP_

#include <limits>
#include <ostream>
#include <random>
#include <vector>

#include "checks.hpp"
#include "clipped_index_strategy.hpp"
#include "index_strategy.hpp"

namespace sacred {

  using std::numeric_limits;
  using std::ostream;
  using std::vector;

  template <typename F, typename IS = ClippedIndexStrategy>
  class Array {
  public:

    Array() = default;

    explicit Array(const vector<int> &shape) : capacity_(), count_(), shape_(shape), index_strategy_() {
      Reshape(shape);
    }

    Array(const vector<int> &shape, const vector<F> &data)
        : capacity_(data.size()), count_(), shape_(shape), data_(data), index_strategy_() {
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
      auto offset = index_strategy_.Offset(data_.size(), shape_, indices);
      if (offset < 0) {
        return F(0.0);
      }
      return data_.at(offset);
    }

    F add(const vector<int> &indices, const F x) {
      auto offset = index_strategy_.Offset(data_.size(), shape_, indices);
      if (offset < 0) {
        return F(0.0);
      }
      data_.at(offset) += x;
      return data_.at(offset);
    }

    F axpby(const vector<int> &indices, const F alpha, const F x, const F beta) {
      auto offset = index_strategy_.Offset(data_.size(), shape_, indices);
      if (offset < 0) {
        return F(0.0);
      }
      data_.at(offset) *= beta;
      data_.at(offset) += alpha * x;
      return data_.at(offset);
    }

    F set(const vector<int> &indices, const F x) {
      auto offset = index_strategy_.Offset(data_.size(), shape_, indices);
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

    void Randomize() {
      std::random_device r;
      std::mt19937 generator(r());
      std::uniform_real_distribution<F> uniform(F(-0.1), F(0.1));
      for (auto i = 0; i < count(); ++i) {
        data(i) = uniform(generator);
      }
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
      if (index_strategy_.Resize() && count_ > capacity_) {
        capacity_ = count_;
        data_.resize(capacity_);
      }
    }

  private:
    int capacity_, count_;
    vector<int> shape_;
    vector<F> data_;
    IS index_strategy_;
  };

  template <typename F>
  ostream &operator <<(ostream &out, const Array<F> &array) {
    if (1 == array.number_of_axes()) {
      for (auto k = 0; k < array.shape(1); ++k) {
        out << array.at({k}) << u8" ";
      }
      out << std::endl;
    } else if (2 == array.number_of_axes()) {
      for (auto j = 0; j < array.shape(0); ++j) {
        for (auto k = 0; k < array.shape(1); ++k) {
          out << array.at({j, k}) << u8" ";
        }
        out << std::endl;
      }
    } else if (3 == array.number_of_axes()) {
      for (auto i = 0; i < array.shape(0); ++i) {
        for (auto j = 0; j < array.shape(1); ++j) {
          for (auto k = 0; k < array.shape(1); ++k) {
            out << array.at({i, j, k}) << u8" ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
    }
    return out;
  }

}  // namespace sacred

#endif  // SACRED_ARRAY_HPP_
