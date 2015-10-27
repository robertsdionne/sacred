#ifndef SACRED_IDENTITY_LOOKUP_HPP_
#define SACRED_IDENTITY_LOOKUP_HPP_

#include "default_types.hpp"
#include "lookup_strategy.hpp"

namespace sacred {

template <typename F = default_floating_point_type, typename I = default_integer_type>
class IdentityLookup : public LookupStrategy<F, I> {
public:
  using storage_type = typename default_storage_type<F>::value;
  using index_type = typename default_index_type<I>::value;

  IdentityLookup() = default;

  virtual F Lookup(const storage_type &data, I data_size,
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    auto offset = I(0);
    for (auto i = I(0); i < stride.size(); ++i) {
      offset += index.at(i) * stride.at(i);
    }
    return data[offset];
  }

  virtual F &Lookup(
      storage_type &data, I data_size,
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    auto offset = I(0);
    for (auto i = I(0); i < stride.size(); ++i) {
      offset += index.at(i) * stride.at(i);
    }
    return data[offset];
  }
};

}  // namespace sacred

#endif  // SACRED_IDENTITY_LOOKUP_HPP_
