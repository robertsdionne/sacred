#ifndef SACRED_INDEX_STRATEGY_HPP_
#define SACRED_INDEX_STRATEGY_HPP_

#include <vector>

#include "default_types.hpp"

namespace sacred {

using std::vector;

// TODO(robertsdionne): Promote tensor namespace to parent.
namespace tensor {

template <typename I>
class IndexStrategy {
public:
  using index_type = typename default_index_type<I>::value;

  virtual ~IndexStrategy() = default;

  virtual index_type Transform(
      const index_type &shape, const index_type &stride, const index_type &index) const = 0;
};

}  // namespace tensor

// TODO(robertsdionne): Remove legacy IndexStrategy.
template <typename F>
class IndexStrategy {
public:
  virtual ~IndexStrategy() = default;

  virtual F Parity(const vector<int> &indices) const {
    return F(1.0);
  }

  virtual bool Resize() const {
    return true;
  };

  /**
   * Calculates the index of data given the data size, array shape and indices.
   * @param  data_size The size of the data buffer.
   * @param  shape     The shape of the tensor.
   * @param  indices   The indices into the tensor.
   * @return           The index into the data buffer corresponding to the given indices, or -1 if out of bounds.
   */
  virtual int Offset(int data_size, const vector<int> &shape, const vector<int> &indices) const = 0;
};

}  // namespace sacred


#endif  // SACRED_INDEX_STRATEGY_HPP_
