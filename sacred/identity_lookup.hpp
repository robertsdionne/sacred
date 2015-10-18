#ifndef SACRED_IDENTITY_LOOKUP_HPP_
#define SACRED_IDENTITY_LOOKUP_HPP_

#include <vector>

#include "lookup_strategy.hpp"

namespace sacred {

  using std::vector;

  template <typename F = float>
  class IdentityLookup : public tensor::LookupStrategy<F> {
  public:
    IdentityLookup() = default;

    virtual int Offset(
        int data_size, const vector<int> &shape, const vector<int> &stride, const vector<int> &index) const override {
      auto result = 0;
      for (auto i = 0; i < stride.size(); ++i) {
        result += index.at(i) * stride.at(i);
      }
      return result;
    }
  };

}  // namespace sacred

#endif  // SACRED_IDENTITY_LOOKUP_HPP_
