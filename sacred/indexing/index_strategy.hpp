#ifndef SACRED_INDEX_STRATEGY_HPP_
#define SACRED_INDEX_STRATEGY_HPP_

#include <vector>

#include "../default_types.hpp"

namespace sacred {

using std::vector;

template <typename I>
class IndexStrategy {
public:
  using index_type = typename default_index_type<I>::value;

  virtual ~IndexStrategy() = default;

  virtual index_type Transform(
      const index_type &shape, const index_type &stride, const index_type &index) const = 0;
};

}  // namespace sacred


#endif  // SACRED_INDEX_STRATEGY_HPP_
