#ifndef SACRED_TILED_INDEX_STRATEGY_HPP_
#define SACRED_TILED_INDEX_STRATEGY_HPP_

#include "index_strategy.hpp"

namespace sacred {

  class TiledIndexStrategy : public IndexStrategy {
  public:
    TiledIndexStrategy() = default;

    virtual ~TiledIndexStrategy() = default;

    int Offset(int data_size, const vector<int> &shape, const vector<int> &indices) const override {
      return -1;
    }
  };

}  // namespace sacred

#endif  // SACRED_TILED_INDEX_STRATEGY_HPP_
