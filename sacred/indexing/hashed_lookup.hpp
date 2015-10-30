#ifndef SACRED_HASHED_LOOKUP_HPP_
#define SACRED_HASHED_LOOKUP_HPP_

#include <boost/functional/hash.hpp>

#include "../default_types.hpp"
#include "lookup_strategy.hpp"

namespace sacred { namespace indexing {

template <typename F = default_floating_point_type, typename I = default_integer_type>
class HashedLookup : public LookupStrategy<F, I> {
public:
  using storage_type = typename default_storage_type<F>::value;
  using index_type = typename default_index_type<I>::value;

  HashedLookup() = default;

  virtual F Lookup(const storage_type &data, I data_size,
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    auto hash_value = hasher(index);
    return Parity(hash_value) * data[hash_value % data_size];
  }

  virtual F &Lookup(storage_type &data, I data_size,
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    auto hash_value = hasher(index);
    return data[hash_value % data_size];
  }

private:
  static constexpr size_t kSeed = 0x7ff83ce;

  F Parity(const size_t index_hash_value) const {
    auto hash_value = kSeed;
    boost::hash_combine(hash_value, index_hash_value);
    return F(1) - F(2) * (hash_value % 2);
  }

private:
  boost::hash<index_type> hasher;
};

}}  // namespaces

#endif  // SACRED_HASHED_LOOKUP_HPP_
