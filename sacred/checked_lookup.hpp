#ifndef SACRED_CHECKED_LOOKUP_HPP_
#define SACRED_CHECKED_LOOKUP_HPP_

#include <vector>

#include "checks.hpp"
#include "default_types.hpp"
#include "identity_lookup.hpp"
#include "lookup_strategy.hpp"

namespace sacred {

  using std::vector;

  template <typename I = default_integer_type>
  class CheckedLookup : public tensor::LookupStrategy<I> {
  public:
    using index_type = typename default_index_type<I>::value;

    CheckedLookup() = default;

    virtual I Offset(
        I data_size, const index_type &shape, const index_type &stride, const index_type &index) const override {
      CHECK_STATE(index.size() <= shape.size());
      for (auto i = I(0); i < shape.size(); ++i) {
        CHECK_STATE(0 <= index.at(i) && index.at(i) < shape.at(i));
      }
      return identity_.Offset(data_size, shape, stride, index);
    }

  private:
    IdentityLookup<I> identity_;
  };

}  // namespace sacred

#endif  // SACRED_CHECKED_LOOKUP_HPP_
