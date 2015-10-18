#ifndef SACRED_WRAPPED_INDEX_HPP_
#define SACRED_WRAPPED_INDEX_HPP_

#include <vector>

#include "checks.hpp"
#include "index_strategy.hpp"

namespace sacred {

  using std::vector;

  class WrappedIndex : public tensor::IndexStrategy {
  public:
    WrappedIndex() = default;

    virtual vector<int> Transform(
        const vector<int> &shape, const vector<int> &stride, const vector<int> &index) const override {
      auto wrapped_index = vector<int>(index.size());
      for (auto i = 0; i < index.size(); ++i) {
        wrapped_index.at(i) = index.at(i) % shape.at(i) + shape.at(i) * (index.at(i) < 0);
      }
      return wrapped_index;
    }
  };

}  // namespace sacred

#endif  // SACRED_WRAPPED_INDEX_HPP_
