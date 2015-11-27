#include <gtest/gtest.h>
#include <utility>

#include "dual.hpp"
#include "gradients.hpp"
#include "smooth_exponential_linear.hpp"
#include "smooth_exponential_linear_gradient.hpp"

namespace sacred {

TEST(SmoothExponentialLinearGradient, Run) {
  auto input = Tensor<>({8}, {-4, -2, -1, -0.5, 0, 1, 2, 3});
  auto input_gradient = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});
  auto op = SmoothExponentialLinearGradient<>(2);
  auto output = Tensor<>({8}, {
    -1.7293295,
    -1.2642411,
    -0.78693867,
    -0.44239843,
    0, 1, 2, 3,
  });
  auto output_gradient = Tensor<>({8}, {
    1, 1, 1, 1, 1, 1, 1, 1,
  });

  op(output, output_gradient, input, input_gradient);

  EXPECT_FLOAT_EQ(0.1353353, input_gradient.at({0}));
  EXPECT_FLOAT_EQ(0.36787948, input_gradient.at({1}));
  EXPECT_FLOAT_EQ(0.60653067, input_gradient.at({2}));
  EXPECT_FLOAT_EQ(0.77880079, input_gradient.at({3}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({4}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({5}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({6}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({7}));
}

TEST(SmoothExponentialLinearGradient, Dual) {
  using std::make_pair;

  auto input = Tensor<Dual>({8}, {-4, -2, -1, -0.5, 0, 1, 2, 3});
  auto target = Tensor<Dual>({8}, {
    -2.7293295,
    -2.2642411,
    -1.78693867,
    -1.44239843,
    -1, 0, 1, 2});

  auto input_gradient = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});

  TestGradients<SmoothExponentialLinear<Dual>>({
    make_pair(&input, &input_gradient),
  }, [] () {
    return new Tensor<Dual>({8}, {0, 0, 0, 0, 0, 0, 0, 0});
  }, [] () {
    return new SmoothExponentialLinear<Dual>(2);
  }, input, target);

  EXPECT_FLOAT_EQ(0.1353353, input_gradient.at({0}));
  EXPECT_FLOAT_EQ(0.36787948, input_gradient.at({1}));
  EXPECT_FLOAT_EQ(0.60653067, input_gradient.at({2}));
  EXPECT_FLOAT_EQ(0.77880079, input_gradient.at({3}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({4}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({5}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({6}));
  EXPECT_FLOAT_EQ(1, input_gradient.at({7}));
}

}  // namespace sacred
