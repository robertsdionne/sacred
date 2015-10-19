#ifndef SACRED_MIRRORED_INDEX_HPP_
#define SACRED_MIRRORED_INDEX_HPP_

#include <vector>

#include "checks.hpp"
#include "index_strategy.hpp"
#include "testing.hpp"

namespace sacred {

using std::vector;

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


class MirroredIndex : public tensor::IndexStrategy {
public:
  MirroredIndex() = default;

  int mod(int x, int y) const {
    return (x % y + y * (x < 0)) % y;
  }

  virtual vector<int> Transform(
      const vector<int> &shape, const vector<int> &stride, const vector<int> &index) const override {
    auto mirrored_index = vector<int>(index.size());
    for (auto i = 0; i < index.size(); ++i) {
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
