#include <gtest/gtest.h>
#include <utility>

#include "dual.hpp"
#include "gradients.hpp"
#include "exponential_linear.hpp"
#include "exponential_linear_gradient.hpp"

namespace sacred {

TEST(ExponentialLinearGradient, Run) {
  auto input = Tensor<>({8}, {-4, -2, -1, -0.5, 0, 1, 2, 3});
  auto input_gradient = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});
  auto op = ExponentialLinearGradient<>();
  auto output = Tensor<>({8}, {
    -0.98168439,
    -0.86466473,
    -0.63212055,
    -0.39346933,
    0, 1, 2, 3,
  });
  auto output_gradient = Tensor<>({8}, {
    1, 1, 1, 1, 1, 1, 1, 1,
  });

  op(output, output_gradient, input, input_gradient);

  EXPECT_FLOAT_EQ(0.018315613, input_gradient.at({0}));
  EXPECT_FLOAT_EQ(0.13533527, input_gradient.at({1}));
  EXPECT_FLOAT_EQ(0.36787945, input_gradient.at({2}));
  EXPECT_FLOAT_EQ(0.60653067, input_gradient.at({3}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({4}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({5}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({6}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({7}));
}

TEST(ExponentialLinearGradient, Dual) {
  using std::make_pair;

  auto input = Tensor<Dual>({8}, {-4, -2, -1, -0.5, 0, 1, 2, 3});
  auto target = Tensor<Dual>({8}, {
    -1.98168439,
    -1.86466473,
    -1.63212055,
    -1.39346933,
    -1, 0, 1, 2});

  auto input_gradient = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});

  TestGradients<ExponentialLinear<Dual>>({
    make_pair(&input, &input_gradient),
  }, [] () {
    return new Tensor<Dual>({8}, {0, 0, 0, 0, 0, 0, 0, 0});
  }, [] () {
    return new ExponentialLinear<Dual>();
  }, input, target);

  EXPECT_FLOAT_EQ(0.018315639, input_gradient.at({0}));
  EXPECT_FLOAT_EQ(0.13533527, input_gradient.at({1}));
  EXPECT_FLOAT_EQ(0.36787945, input_gradient.at({2}));
  EXPECT_FLOAT_EQ(0.60653067, input_gradient.at({3}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({4}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({5}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({6}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({7}));
}

}  // namespace sacred
