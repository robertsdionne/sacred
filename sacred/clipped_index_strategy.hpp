#ifndef SACRED_CLIPPING_INDEXING_STRATEGY_HPP_
#define SACRED_CLIPPING_INDEXING_STRATEGY_HPP_

#include "checks.hpp"
#include "index_strategy.hpp"

namespace sacred {

  template <typename F>
  class ClippedIndexStrategy : public IndexStrategy<F> {
  public:
    ClippedIndexStrategy() = default;

    virtual ~ClippedIndexStrategy() = default;

    int Offset(int data_size, const vector<int> &shape, const vector<int> &indices) const override {
      CHECK_STATE(indices.size() == shape.size());
      int offset = 0;
      for (auto i = 0; i < shape.size(); ++i) {
        offset *= shape.at(i);
        if (indices.size() > i) {
          if (0 <= indices.at(i) && indices.at(i) < shape.at(i)) {
            offset += indices.at(i);
          } else {
            return -1;
          }
        }
      }
      return offset;
    }
  };

}  // namespace sacred

#endif  // SACRED_CLIPPING_INDEXING_STRATEGY_HPP_
