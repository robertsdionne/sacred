#ifndef SACRED_MIRRORED_INDEX_HPP_
#define SACRED_MIRRORED_INDEX_HPP_

#include "checks.hpp"
#include "default_types.hpp"
#include "index_strategy.hpp"
#include "testing.hpp"

namespace sacred {

// 01 23 45 67 89
// 01 01 01 01 01
// ++ -- ++ -- ++
// 01 21 01 21 01
//
// 0123456789
// 0122100122
//
// 1232123212
// 1233211233


template <typename I = default_integer_type>
class MirroredIndex : public tensor::IndexStrategy<I> {
public:
  using index_type = typename default_index_type<I>::value;

  MirroredIndex() = default;

  I mod(I x, I y) const {
    return (x % y + y * (x < I(0))) % y;
  }

  virtual index_type Transform(
      const index_type &shape, const index_type &stride, const index_type &index) const override {
    auto mirrored_index = index_type(index.size());
    for (auto i = I(0); i < index.size(); ++i) {
      auto s = shape.at(i) - 1;

      // TODO(robertsdionne): Handle division by 0 here when shape.at(i) = 1.
      auto sign = 1 - 2 * mod(index.at(i) / s, 2);
      mirrored_index.at(i) = sign > 0?
          mod(index.at(i), s):
          s - mod(index.at(i), s);
    }
    return mirrored_index;
  }
};

}  // namespace sacred

#endif  // SACRED_MIRRORED_INDEX_HPP_
