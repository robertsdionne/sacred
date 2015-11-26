#include <gtest/gtest.h>

#include <utility>

#include "dual.hpp"
#include "gradients.hpp"
#include "softmax.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(Softmax, Run) {
  auto input = Tensor<>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto op = Softmax<>();
  auto output = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});

  op(input, output);

  EXPECT_FLOAT_EQ(0.00057661271, output.at({0}));
  EXPECT_FLOAT_EQ(0.0015673959, output.at({1}));
  EXPECT_FLOAT_EQ(0.0042606238, output.at({2}));
  EXPECT_FLOAT_EQ(0.011581576, output.at({3}));
  EXPECT_FLOAT_EQ(0.031481985, output.at({4}));
  EXPECT_FLOAT_EQ(0.0855769, output.at({5}));
  EXPECT_FLOAT_EQ(0.23262219, output.at({6}));
  EXPECT_FLOAT_EQ(0.63233268, output.at({7}));
}

TEST(Softmax, Gradient) {
  using std::make_pair;

  auto input = Tensor<Dual>({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  auto target = Tensor<Dual>({8}, {0, 0, 1, 0, 0, 0, 0, 0});

  auto input_gradient = Tensor<>({8}, {0, 0, 0, 0, 0, 0, 0, 0});

  TestGradients<Softmax<Dual>>({
    make_pair(&input, &input_gradient),
  }, [] () {
    return new Tensor<Dual>({8}, {0, 0, 0, 0, 0, 0, 0, 0});
  }, [] () {
    return new Softmax<Dual>();
  }, input, target);

  EXPECT_FLOAT_EQ(-0.00026385227, input_gradient.at({0}));
  EXPECT_FLOAT_EQ(-0.00071567186, input_gradient.at({1}));
  EXPECT_FLOAT_EQ(-0.0061945468, input_gradient.at({2}));
  EXPECT_FLOAT_EQ(-0.00517216, input_gradient.at({3}));
  EXPECT_FLOAT_EQ(-0.013432883, input_gradient.at({4}));
  EXPECT_FLOAT_EQ(-0.031885084, input_gradient.at({5}));
  EXPECT_FLOAT_EQ(-0.052466642, input_gradient.at({6}));
  EXPECT_FLOAT_EQ(0.11013085, input_gradient.at({7}));
}

} // namespace sacred
