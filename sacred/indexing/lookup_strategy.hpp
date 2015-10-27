#ifndef SACRED_LOOKUP_STRATEGY_HPP_
#define SACRED_LOOKUP_STRATEGY_HPP_

#include "../default_types.hpp"

namespace sacred {

template <typename F, typename I>
class LookupStrategy {
public:
  using storage_type = typename default_storage_type<F>::value;
  using index_type = typename default_index_type<I>::value;

  virtual ~LookupStrategy() = default;

  /**
   * Calculates the index of data given the data size, array shape and indices.
   * @param  data_size The size of the data buffer.
   * @param  shape     The shape of the tensor.
   * @param  stride    The shape of the tensor.
   * @param  index     The indices into the tensor.
   * @return           The index into the data buffer corresponding to the given indices, or -1 if out of bounds.
   */
  virtual F Lookup(
      const storage_type &data, I data_size,
      const index_type &shape, const index_type &stride, const index_type &index) const = 0;

  virtual F &Lookup(
      storage_type &data, I data_size,
      const index_type &shape, const index_type &stride, const index_type &index) const = 0;
};

}  // namespace sacred


#endif  // SACRED_LOOKUP_STRATEGY_HPP_
