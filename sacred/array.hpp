#ifndef SACRED_ARRAY_HPP_
#define SACRED_ARRAY_HPP_

#include <vector>

namespace sacred {

  using std::vector;

  template <typename F>
  class Array {
  public:
    Array(const vector<int> &shape) : shape_(shape) {}

    ~Array() = default;

    const F *data() const {
      return data_;
    }

    F *data() {
      return data_;
    }

    const vector<int> &shape() {
      return shape_;
    }

    void Allocate() {
      if (data_) {
        delete[] data_;
      }
      int size = 1;
      for (auto i = 0; i < shape_.size(); ++i) {
        size *= shape_.at(i);
      }
      data_ = new F[size]();
    }

    const F &at(const vector<int> &indices) const {
      return data()[Offset(indices)];
    }

    F &at(const vector<int> &indices) {
      return data()[Offset(indices)];
    }

    int Offset(const vector<int> &indices) const {
      // TODO(robertsdionne): add some index checks.
      int offset = 0;
      for (auto i = 0; i < shape_.size(); ++i) {
        offset *= shape_.at(i);
        if (indices.size() > i) {
          offset += indices.at(i);
        }
      }
      return offset;
    }

  private:
    vector<int> shape_;
    F *data_;
  };

}  // namespace sacred

#endif  // SACRED_ARRAY_HPP_
