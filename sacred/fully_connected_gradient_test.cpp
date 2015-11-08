#include <gtest/gtest.h>

#include "fully_connected_gradient.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(FullyConnectedGradient, Run) {
  auto input = Tensor<>({4, 1}, {1, 2, 3, 4});
  auto input_gradient = Tensor<>({4, 1}, {0, 0, 0, 0});
  auto bias_gradient = Tensor<>({3, 1}, {0, 0, 0});
  auto weight = Tensor<>({3, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
  });
  auto weight_gradient = Tensor<>({3, 4}, {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  });
  auto op = FullyConnectedGradient<>(bias_gradient, weight, weight_gradient);
  auto output_gradient = Tensor<>({3, 1}, {1, 2, 3});

  op(output_gradient, input, input_gradient);

  EXPECT_EQ(Tensor<>({4, 1}, {38, 44, 50, 56}), input_gradient);
  EXPECT_EQ(Tensor<>({3, 4}, {
    1, 2, 3, 4,
    2, 4, 6, 8,
    3, 6, 9, 12,
  }), weight_gradient);
  EXPECT_EQ(Tensor<>({3, 1}, {1, 2, 3}), bias_gradient);
}

}  // namespace sacred
