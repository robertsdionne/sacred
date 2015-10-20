#ifndef SACRED_HASHED_LOOKUP_HPP_
#define SACRED_HASHED_LOOKUP_HPP_

#include <boost/functional/hash.hpp>

#include "default_types.hpp"
#include "lookup_strategy.hpp"

namespace sacred {

template <typename I = default_integer_type>
class HashedLookup : public tensor::LookupStrategy<I> {
public:
  using index_type = typename default_index_type<I>::value;

  HashedLookup() = default;

  virtual I Offset(
      I data_size, const index_type &shape, const index_type &stride, const index_type &index) const override {
    auto hashed_index = hasher(index) % data_size;
    return hashed_index;
  }

private:
  boost::hash<index_type> hasher;
};

}  // namespace sacred

#endif  // SACRED_HASHED_LOOKUP_HPP_
