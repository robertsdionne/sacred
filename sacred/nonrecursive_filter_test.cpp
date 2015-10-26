#include <gtest/gtest.h>

#include "nonrecursive_filter.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(NonrecursiveFilter, Run) {
  auto input = Tensor<>({4, 4}, {
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7
  });
  auto bias = Tensor<>({2, 2});
  auto filter = Tensor<>({3, 3}, {
    1, 2, 3,
    2, 3, 4,
    3, 4, 5
  });
  auto op = NonrecursiveFilter<>(bias, filter);
  auto output = Tensor<>({2, 2});
  op.Run({&input}, {&output});
  EXPECT_EQ(
      Tensor<>({2, 2}, {
        69, 96,
        96, 123
       }),
      output);
}

}  // namespace sacred
