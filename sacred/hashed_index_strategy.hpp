#ifndef SACRED_HASHED_INDEX_STRATEGY_HPP_
#define SACRED_HASHED_INDEX_STRATEGY_HPP_

#include <boost/functional/hash.hpp>
#include <vector>

#include "checks.hpp"
#include "index_strategy.hpp"

namespace sacred {

using std::vector;

template <typename F>
class HashedIndexStrategy : public IndexStrategy<F> {
public:
  HashedIndexStrategy() = default;

  virtual ~HashedIndexStrategy() = default;

  static constexpr size_t kSeed = 0x7ff83ce;

  F Parity(const vector<int> &indices) const override {
    auto hash = kSeed;
    boost::hash_combine(hash, boost::hash_value(indices));
    return F(1.0) - (hash % 2) * F(2.0);
  }

  bool Resize() const override {
    return false;
  }

  int Offset(int data_size, const vector<int> &shape, const vector<int> &indices) const override {
    return boost::hash_value(indices) % data_size;
  }
};

}  // namespace sacred

#endif  // SACRED_HASHED_INDEX_STRATEGY_HPP_
