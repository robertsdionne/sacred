#ifndef SACRED_FUNCTIONAL_HPP_
#define SACRED_FUNCTIONAL_HPP_

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

namespace sacred {

  using std::accumulate;
  using std::begin;
  using std::end;
  using std::transform;
  using std::vector;

  template <typename F>
  vector<F> SelectFrom(const vector<F> &elements, const vector<int> &indices) {
    vector<F> selection(indices.size());
    transform(begin(indices), end(indices), begin(selection), [&elements] (int index) {
      return elements.at(index);
    });
    return selection;
  }

  template <typename F>
  F ProductOf(const vector<F> &elements) {
    return accumulate(begin(elements), end(elements), F(1.0), [] (F left, F right) {
      return left * right;
    });
  }

}  // namespace sacred

#endif  // SACRED_FUNCTIONAL_HPP_
