#ifndef SACRED_MASKED_LOOKUP_HPP_
#define SACRED_MASKED_LOOKUP_HPP_

#include <limits>

#include "../default_types.hpp"
#include "identity_lookup.hpp"
#include "lookup_strategy.hpp"

namespace sacred {

using std::numeric_limits;

enum class MaskDefault {
  kZero,
  kNaN
};

template <typename F = default_floating_point_type,
    typename I = default_integer_type, MaskDefault mask_default = MaskDefault::kZero>
class MaskedLookup : public LookupStrategy<F, I> {
public:
  using storage_type = typename default_storage_type<F>::value;
  using index_type = typename default_index_type<I>::value;

  MaskedLookup() = default;

  virtual F Lookup(const storage_type &data, I data_size,
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    for (auto i = I(0); i < shape.size(); ++i) {
      if (index.at(i) < 0 || shape.at(i) <= index.at(i)) {
        return DefaultValue();
      }
    }
    return identity_.Lookup(data, data_size, shape, stride, index);
  }

  virtual F &Lookup(storage_type &data, I data_size,
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    for (auto i = I(0); i < shape.size(); ++i) {
      if (index.at(i) < 0 || shape.at(i) <= index.at(i)) {
        return identity_.Lookup(data, data_size, shape, stride, index);
      }
    }
    return identity_.Lookup(data, data_size, shape, stride, index);
  }

private:
  F DefaultValue() const {
    return MaskDefault::kZero == mask_default ? F(0) : numeric_limits<F>::quiet_NaN();
  }

private:
  IdentityLookup<F, I> identity_;
};

}  // namespace sacred

#endif  // SACRED_MASKED_LOOKUP_HPP_
