#ifndef SACRED_IDENTITY_INDEX_HPP_
#define SACRED_IDENTITY_INDEX_HPP_

#include "default_types.hpp"
#include "index_strategy.hpp"

namespace sacred {

template <typename I = default_integer_type>
class IdentityIndex : public tensor::IndexStrategy<I> {
public:
  using index_type = typename default_index_type<I>::value;

  IdentityIndex() = default;

  virtual index_type Transform(
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    return index;
  }
};

}  // namespace sacred

#endif  // SACRED_IDENTITY_INDEX_HPP_