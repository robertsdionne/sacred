#include <gtest/gtest.h>

#include "softmax_gradient.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(SoftmaxGradient, Run) {
  auto input = Tensor<>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto input_gradient = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});
  auto op = SoftmaxGradient<>();
  auto output = Tensor<>({8}, {
    0.00057661271,
    0.0015673959,
    0.0042606238,
    0.011581576,
    0.031481985,
    0.0855769,
    0.23262219,
    0.63233268,
  });
  auto output_gradient = Tensor<>({8}, {
    0.00057661271 - 0,
    0.0015673959 - 0,
    0.0042606238 - 1,
    0.011581576 - 0,
    0.031481985 - 0,
    0.0855769 - 0,
    0.23262219 - 0,
    0.63233268 - 0,
  });

  op(output_gradient, output, input_gradient);

  EXPECT_FLOAT_EQ(-0.00026385227, input_gradient.at({0}));
  EXPECT_FLOAT_EQ(-0.00071567186, input_gradient.at({1}));
  EXPECT_FLOAT_EQ(-0.0061945468, input_gradient.at({2}));
  EXPECT_FLOAT_EQ(-0.00517216, input_gradient.at({3}));
  EXPECT_FLOAT_EQ(-0.013432883, input_gradient.at({4}));
  EXPECT_FLOAT_EQ(-0.031885084, input_gradient.at({5}));
  EXPECT_FLOAT_EQ(-0.052466642, input_gradient.at({6}));
  EXPECT_FLOAT_EQ(0.11013085, input_gradient.at({7}));
}

}  // namespace sacred
