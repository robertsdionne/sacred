#ifndef SACRED_WRAPPED_INDEX_HPP_
#define SACRED_WRAPPED_INDEX_HPP_

#include <vector>

#include "checks.hpp"
#include "default_types.hpp"
#include "index_strategy.hpp"

namespace sacred {

  using std::vector;

  template <typename I = default_integer_type>
  class WrappedIndex : public tensor::IndexStrategy<I> {
  public:
    using index_type = typename default_index_type<I>::value;

    WrappedIndex() = default;

    virtual index_type Transform(
        const index_type &shape, const index_type &stride, const index_type &index) const override {
      auto wrapped_index = index_type(index.size());
      for (auto i = I(0); i < index.size(); ++i) {
        wrapped_index.at(i) = index.at(i) % shape.at(i) + shape.at(i) * (index.at(i) < 0);
      }
      return wrapped_index;
    }
  };

}  // namespace sacred

#endif  // SACRED_WRAPPED_INDEX_HPP_
