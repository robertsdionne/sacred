#include <gtest/gtest.h>

#include "fully_connected.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(FullyConnected, Run) {
  auto input = Tensor<>({4, 1}, {1, 2, 3, 4});
  auto bias = Tensor<>({3, 1}, {1, 2, 3});
  auto weight = Tensor<>({3, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
  });
  auto op = FullyConnected<>(bias, weight);
  auto output = Tensor<>({3, 1}, {0, 0, 0});

  op(input, output);

  EXPECT_EQ(Tensor<>({3, 1}, {31, 72, 113}), output);
}

}  // namespace sacred
