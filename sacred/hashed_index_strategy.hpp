#ifndef SACRED_HASHED_INDEX_STRATEGY_HPP_
#define SACRED_HASHED_INDEX_STRATEGY_HPP_

#include <boost/functional/hash.hpp>
#include <vector>

#include "checks.hpp"
#include "index_strategy.hpp"

namespace sacred {

  using std::vector;

  class HashedIndexStrategy : public IndexStrategy {
  public:
    HashedIndexStrategy() = default;

    virtual ~HashedIndexStrategy() = default;

    bool Resize() const override {
      return false;
    }

    int Offset(int data_size, const vector<int> &shape, const vector<int> &indices) const override {
      return boost::hash_value(indices) % data_size;
    }
  };

}  // namespace sacred

#endif  // SACRED_HASHED_INDEX_STRATEGY_HPP_
