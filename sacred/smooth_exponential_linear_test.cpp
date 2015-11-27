#include <gtest/gtest.h>

#include "smooth_exponential_linear.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(SmoothExponentialLinear, Run) {
  auto input = Tensor<>({8}, {-4, -2, -1, -0.5, 0, 1, 2, 3});
  auto op = SmoothExponentialLinear<>(2);
  auto output = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});

  op(input, output);

  EXPECT_FLOAT_EQ(-1.7293295, output.at({0}));
  EXPECT_FLOAT_EQ(-1.2642411, output.at({1}));
  EXPECT_FLOAT_EQ(-0.78693867, output.at({2}));
  EXPECT_FLOAT_EQ(-0.44239843, output.at({3}));
  EXPECT_FLOAT_EQ(0, output.at({4}));
  EXPECT_FLOAT_EQ(1, output.at({5}));
  EXPECT_FLOAT_EQ(2, output.at({6}));
  EXPECT_FLOAT_EQ(3, output.at({7}));
}

}  // namespace sacred
