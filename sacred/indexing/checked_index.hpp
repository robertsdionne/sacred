#ifndef SACRED_CHECKED_INDEX_HPP_
#define SACRED_CHECKED_INDEX_HPP_

#include <glog/logging.h>

#include "../default_types.hpp"
#include "identity_index.hpp"
#include "index_strategy.hpp"

namespace sacred { namespace indexing {

template <typename I = default_integer_type>
class CheckedIndex : public IndexStrategy<I> {
public:
  using index_type = typename default_index_type<I>::value;

  CheckedIndex() = default;

  virtual index_type Transform(
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    CHECK(index.size() <= shape.size());
    for (auto i = I(0); i < shape.size(); ++i) {
      CHECK(0 <= index.at(i) && index.at(i) < shape.at(i));
    }
    return identity_.Transform(shape, stride, index);
  }

private:
  IdentityIndex<I> identity_;
};

}}  // namespaces

#endif  // SACRED_CHECKED_INDEX_HPP_
