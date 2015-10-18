#ifndef SACRED_IDENTITY_INDEX_HPP_
#define SACRED_IDENTITY_INDEX_HPP_

#include "index_strategy.hpp"

namespace sacred {

  class IdentityIndex : public tensor::IndexStrategy {
  public:
    IdentityIndex() = default;

    virtual vector<int> Transform(
        const vector<int> &shape, const vector<int> &stride, const vector<int> &index) const override {
      return index;
    }
  };

}  // namespace sacred

#endif  // SACRED_IDENTITY_INDEX_HPP_
