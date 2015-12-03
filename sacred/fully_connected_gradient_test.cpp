#include <gtest/gtest.h>
#include <utility>

#include "dual.hpp"
#include "fully_connected.hpp"
#include "fully_connected_gradient.hpp"
#include "gradients.hpp"
#include "tensor.hpp"
#include "tensor_testing.hpp"

namespace sacred {

TEST(FullyConnectedGradient, Run) {
  auto input = MakeTestTensor<>({1, 1, 4});
  auto input_gradient = Tensor<>({1, 1, 4});
  auto bias_gradient = Tensor<>({1, 3});
  auto weight = MakeTestTensor<>({1, 3, 1, 4});
  auto weight_gradient = Tensor<>({1, 3, 1, 4});
  auto op = FullyConnectedGradient<>(bias_gradient, weight, weight_gradient);
  auto output_gradient = Tensor<>({1, 1, 3}, {1, 2, 3});

  op(output_gradient, input, input_gradient);

  EXPECT_EQ(Tensor<>({1, 1, 4}, {14, 20, 26, 32}), input_gradient);
  EXPECT_EQ(Tensor<>({1, 3}, {1, 2, 3}), bias_gradient);
  EXPECT_EQ(Tensor<>({1, 3, 1, 4}, {
    1, 2, 3, 4,
    2, 4, 6, 8,
    3, 6, 9, 12,
  }), weight_gradient);
}

TEST(FullyConnectedGradient, Dual) {
  using std::make_pair;

  auto input = MakeTestTensor<Dual>({1, 1, 4});
  auto bias = MakeTestTensor<Dual>({1, 3});
  auto weight = MakeTestTensor<Dual>({1, 3, 1, 4});
  auto target = Tensor<Dual>({1, 1, 3}, {30, 40, 50});

  auto input_gradient = Tensor<>({1, 1, 4});
  auto bias_gradient = Tensor<>({1, 3});
  auto weight_gradient = Tensor<>({1, 3, 1, 4});

  TestGradients<FullyConnected<Dual>>({
    make_pair(&input, &input_gradient),
    make_pair(&bias, &bias_gradient),
    make_pair(&weight, &weight_gradient),
  }, [] () {
    return new Tensor<Dual>({1, 1, 3}, {0, 0, 0});
  }, [&bias, &weight] () {
    return new FullyConnected<Dual>(bias, weight);
  }, input, target);

  EXPECT_EQ(Tensor<>({1, 1, 4}, {14, 20, 26, 32}), input_gradient);
  EXPECT_EQ(Tensor<>({1, 3}, {1, 2, 3}), bias_gradient);
  EXPECT_EQ(Tensor<>({1, 3, 1, 4}, {
    1, 2, 3, 4,
    2, 4, 6, 8,
    3, 6, 9, 12,
  }), weight_gradient);
}

}  // namespace sacred
