#ifndef SACRED_TENSOR_TESTING_HPP_
#define SACRED_TENSOR_TESTING_HPP_

#include <numeric>

#include "default_types.hpp"
#include "tensor.hpp"

namespace sacred {

template <typename F=default_floating_point_type, typename I=default_integer_type>
Tensor<F, I> MakeTestTensor(const typename default_index_type<I>::value &shape) {
  using std::accumulate;

  auto test_tensor = Tensor<F, I>(shape);
  for (auto i = 0; i < test_tensor.size(); ++i) {
    auto index = test_tensor.index(i);
    auto sum = accumulate(index.begin(), index.end(), F(1));
    test_tensor.data(i) = sum;
  }
  return test_tensor;
}

}

#endif  // SACRED_TENSOR_TESTING_HPP_
