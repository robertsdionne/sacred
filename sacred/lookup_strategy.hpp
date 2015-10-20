#ifndef SACRED_LOOKUP_STRATEGY_HPP_
#define SACRED_LOOKUP_STRATEGY_HPP_

#include <vector>

namespace sacred {

// TODO(robertsdionne): Promote tensor namespace to parent.
namespace tensor {

using std::vector;

template <typename I>
class LookupStrategy {
public:
  using index_type = vector<I>;

  virtual ~LookupStrategy() = default;

  /**
   * Calculates the index of data given the data size, array shape and indices.
   * @param  data_size The size of the data buffer.
   * @param  shape     The shape of the tensor.
   * @param  stride    The shape of the tensor.
   * @param  index     The indices into the tensor.
   * @return           The index into the data buffer corresponding to the given indices, or -1 if out of bounds.
   */
  virtual I Offset(
      I data_size, const index_type &shape, const index_type &stride, const index_type &index) const = 0;
};

}  // namespace tensor

}  // namespace sacred


#endif  // SACRED_LOOKUP_STRATEGY_HPP_
