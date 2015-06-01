#ifndef SACRED_LOOKUP_TABLE_HPP_
#define SACRED_LOOKUP_TABLE_HPP_

#include <string>

#include "blob.hpp"

namespace sacred {

  using std::string;

  template <typename F>
  class LookupTable {
  public:
    virtual ~LookupTable() = default;

    virtual void Forward(const vector<string> &bottom, Blob<F> *top) = 0;

    virtual void Backward(const Blob<F> &top, const vector<string> &bottom) = 0;
  };

}  // namespace sacred

#endif  // SACRED_LOOKUP_TABLE_HPP_
