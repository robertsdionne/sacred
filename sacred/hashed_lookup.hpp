#ifndef SACRED_HASHED_LOOKUP_HPP_
#define SACRED_HASHED_LOOKUP_HPP_

#include "default_types.hpp"
#include "lookup_strategy.hpp"

namespace sacred {

template <typename I = default_integer_type>
class HashedLookup : public tensor::LookupStrategy<I> {
public:
  using index_type = typename default_index_type<I>::value;

  HashedLookup() = default;

  virtual index_type Transform(
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    auto hashed_index = 0;
    return hashed_index;
  }
};

}  // namespace sacred

#endif  // SACRED_HASHED_LOOKUP_HPP_
