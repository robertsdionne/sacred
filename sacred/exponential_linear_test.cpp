#include <gtest/gtest.h>

#include <utility>

#include "dual.hpp"
#include "gradients.hpp"
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

TEST(ExponentialLinear, Gradient) {
  using std::make_pair;

  auto input = Tensor<Dual>({8}, {-4, -3, -2, -1, 0, 1, 2, 3});
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
  EXPECT_FLOAT_EQ(0.045527868, input_gradient.at({1}));
  EXPECT_FLOAT_EQ(0.10386386, input_gradient.at({2}));
  EXPECT_FLOAT_EQ(0.28008458, input_gradient.at({3}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({4}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({5}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({6}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({7}));
}

}  // namespace sacred
