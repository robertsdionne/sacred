#include <gtest/gtest.h>
#include <iostream>

#include "recursive_filter.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(RecursiveFilter, Run) {
  auto input = Tensor<>({4, 4}, {
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7
  });
  auto bias = Tensor<>({4}, {0, 0, 0, 0});
  auto filter = Tensor<>({3, 3}, {
    1, 2, 3,
    2, 3, 4,
    4, 5, 6
  });
  auto op = RecursiveFilter<>(bias, filter);
  auto output = Tensor<>({4, 4});
  op({&input}, {&output});
  // EXPECT_EQ(
  //     Tensor<>({4, 4}, {
  //       1, 6, 36, 227,
  //       2, 14, 95, 635,
  //       3, 22, 157, 1093,
  //       4, 25, 171, 1196
  //     }),
  //     output);
}

}  // namespace sacred
