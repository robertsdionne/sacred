#ifndef SACRED_LOOKUP_STRATEGY_HPP_
#define SACRED_LOOKUP_STRATEGY_HPP_

#include <vector>

namespace sacred {

// TODO(robertsdionne): Promote tensor namespace to parent.
namespace tensor {

using std::vector;

template <typename F>
class LookupStrategy {
public:
  virtual ~LookupStrategy() = default;

  /**
   * Calculates the index of data given the data size, array shape and indices.
   * @param  data_size The size of the data buffer.
   * @param  shape     The shape of the tensor.
   * @param  stride    The shape of the tensor.
   * @param  index     The indices into the tensor.
   * @return           The index into the data buffer corresponding to the given indices, or -1 if out of bounds.
   */
  virtual int Offset(
      int data_size, const vector<int> &shape, const vector<int> &stride, const vector<int> &index) const = 0;
};

}  // namespace tensor

}  // namespace sacred


#endif  // SACRED_LOOKUP_STRATEGY_HPP_
