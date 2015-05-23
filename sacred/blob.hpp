#ifndef SACRED_BLOB_HPP_
#define SACRED_BLOB_HPP_

#include <vector>

#include "array.hpp"

namespace sacred {

  using std::vector;

  template <typename F>
  class Blob {
  public:
    Blob(const vector<int> &shape) : value_(shape), diff_(shape) {}

    ~Blob() = default;

    const Array<F> &diff() const {
      return diff_;
    }

    Array<F> &diff() {
      return diff_;
    }

    const vector<int> &shape() {
      return value_.shape();
    }

    const Array<F> &value() const {
      return value_;
    }

    Array<F> &value() {
      return value_;
    }

    void Allocate() {
      value_.Allocate();
      diff_.Allocate();
    }

  private:
    Array<F> value_, diff_;
  };

}  // namespace sacred

#endif  // SACRED_BLOB_HPP_
