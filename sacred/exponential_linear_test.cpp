#include <gtest/gtest.h>

#include "exponential_linear.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(ExponentialLinear, Run) {
  auto input = Tensor<>({8}, {-4, -2, -1, -0.5, 0, 1, 2, 3});
  auto op = ExponentialLinear<>();
  auto output = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});

  op(input, output);

  EXPECT_FLOAT_EQ(-0.98168439, output.at({0}));
  EXPECT_FLOAT_EQ(-0.86466473, output.at({1}));
  EXPECT_FLOAT_EQ(-0.63212055, output.at({2}));
  EXPECT_FLOAT_EQ(-0.39346933, output.at({3}));
  EXPECT_FLOAT_EQ(0, output.at({4}));
  EXPECT_FLOAT_EQ(1, output.at({5}));
  EXPECT_FLOAT_EQ(2, output.at({6}));
  EXPECT_FLOAT_EQ(3, output.at({7}));
}

}  // namespace sacred
