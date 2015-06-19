#ifndef SACRED_TILED_INDEX_STRATEGY_HPP_
#define SACRED_TILED_INDEX_STRATEGY_HPP_

#include <iostream>

#include "checks.hpp"
#include "index_strategy.hpp"

namespace sacred {

  class TiledIndexStrategy : public IndexStrategy {
  public:
    TiledIndexStrategy() = default;

    virtual ~TiledIndexStrategy() = default;

    int Offset(int data_size, const vector<int> &shape, const vector<int> &indices) const override {
      CHECK_STATE(indices.size() == shape.size());
      int offset = 0;
      for (auto i = 0; i < shape.size(); ++i) {
        offset *= shape.at(i);
        if (indices.size() > i) {
          auto index = indices.at(i) % shape.at(i);
          index += (index < 0) * shape.at(i);
          offset += index;
        }
      }
      std::cout << std::endl;
      return offset;
    }
  };

}  // namespace sacred

#endif  // SACRED_TILED_INDEX_STRATEGY_HPP_
