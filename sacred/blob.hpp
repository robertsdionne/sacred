#ifndef SACRED_BLOB_HPP_
#define SACRED_BLOB_HPP_

#include <vector>

#include "array.hpp"

namespace sacred {

  using std::vector;

  template <typename F>
  class Blob {
  public:
    Blob() = default;

    Blob(const vector<int> &shape) : value_(shape), diff_(shape) {}

    ~Blob() = default;

    inline int count() const {
      return value_.count();
    }

    inline const Array<F> &diff() const {
      return diff_;
    }

    inline Array<F> &diff() {
      return diff_;
    }

    inline int number_of_axes() const {
      return value_.number_of_axes();
    }

    inline const vector<int> &shape() const {
      return value_.shape();
    }

    inline int shape(int index) const {
      return value_.shape(index);
    }

    inline const Array<F> &value() const {
      return value_;
    }

    inline Array<F> &value() {
      return value_;
    }

    void Reshape(const vector<int> &shape) {
      value_.Reshape(shape);
      diff_.Reshape(shape);
    }

  private:
    Array<F> value_, diff_;
  };

}  // namespace sacred

#endif  // SACRED_BLOB_HPP_
