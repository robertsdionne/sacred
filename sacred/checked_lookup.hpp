#ifndef SACRED_CHECKED_LOOKUP_HPP_
#define SACRED_CHECKED_LOOKUP_HPP_

#include <vector>

#include "checks.hpp"
#include "identity_lookup.hpp"
#include "lookup_strategy.hpp"

namespace sacred {

  using std::vector;

  class CheckedLookup : public tensor::LookupStrategy {
  public:
    CheckedLookup() = default;

    virtual int Offset(
        int data_size, const vector<int> &shape, const vector<int> &stride, const vector<int> &index) const override {
      CHECK_STATE(index.size() <= shape.size());
      for (auto i = 0; i < shape.size(); ++i) {
        CHECK_STATE(0 <= index.at(i) && index.at(i) < shape.at(i));
      }
      return identity_.Offset(data_size, shape, stride, index);
    }

  private:
    IdentityLookup identity_;
  };

}  // namespace sacred

#endif  // SACRED_CHECKED_LOOKUP_HPP_
