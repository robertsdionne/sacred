#include <gtest/gtest.h>
#include <utility>

#include "dual.hpp"
#include "fully_connected.hpp"
#include "fully_connected_gradient.hpp"
#include "gradients.hpp"
#include "tensor.hpp"
#include "tensor_testing.hpp"

namespace sacred {

TEST(FullyConnectedGradient, RunSimple) {
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

TEST(FullyConnectedGradient, DualSimple) {
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

TEST(FullyConnectedGradient, Run) {
  auto input = MakeTestTensor<>({4, 2, 4});
  auto input_gradient = Tensor<>({4, 2, 4});
  auto bias_gradient = Tensor<>({3, 3});
  auto weight = MakeTestTensor<>({3, 3, 2, 4});
  auto weight_gradient = Tensor<>({3, 3, 2, 4});
  auto op = FullyConnectedGradient<>(bias_gradient, weight, weight_gradient);
  auto output_gradient = MakeTestTensor<>({4, 3, 3});

  op(output_gradient, input, input_gradient);

  EXPECT_EQ(Tensor<>({4, 2, 4}, {
    93, 120, 147, 174,   120, 147, 174, 201,
    120, 156, 192, 228,  156, 192, 228, 264,
    147, 192, 237, 282,  192, 237, 282, 327,
    174, 228, 282, 336,  228, 282, 336, 390,
  }), input_gradient);
  EXPECT_EQ(Tensor<>({3, 3}, {
    10, 14, 18,
    14, 18, 22,
    18, 22, 26,
  }), bias_gradient);
  EXPECT_EQ(Tensor<>({3, 3, 2, 4}, {
    30, 40, 50, 60,    40, 50, 60, 70,
    40, 54, 68, 82,    54, 68, 82, 96,
    50, 68, 86, 104,   68, 86, 104, 122,

    40, 54, 68, 82,    54, 68, 82, 96,
    50, 68, 86, 104,   68, 86, 104, 122,
    60, 82, 104, 126,  82, 104, 126, 148,

    50, 68, 86, 104,   68, 86, 104, 122,
    60, 82, 104, 126,  82, 104, 126, 148,
    70, 96, 122, 148,  96, 122, 148, 174
  }), weight_gradient);
}

TEST(FullyConnectedGradient, Dual) {
  using std::make_pair;

  auto input = MakeTestTensor<Dual>({4, 2, 4});
  auto bias = MakeTestTensor<Dual>({3, 3});
  auto weight = MakeTestTensor<Dual>({3, 3, 2, 4});
  auto target = Tensor<Dual>({4, 3, 3}, {
    85 - 1, 110 - 2, 135 - 3,
    110 - 2, 135 - 3, 160 - 4,
    135 - 3, 160 - 4, 185 - 5,

    109 - 2, 142 - 3, 175 - 4,
    142 - 3, 175 - 4, 208 - 5,
    175 - 4, 208 - 5, 241 - 6,

    133 - 3, 174 - 4, 215 - 5,
    174 - 4, 215 - 5, 256 - 6,
    215 - 5, 256 - 6, 297 - 7,

    157 - 4, 206 - 5, 255 - 6,
    206 - 5, 255 - 6, 304 - 7,
    255 - 6, 304 - 7, 353 - 8,
  });

  auto input_gradient = Tensor<>({4, 2, 4});
  auto bias_gradient = Tensor<>({3, 3});
  auto weight_gradient = Tensor<>({3, 3, 2, 4});

  TestGradients<FullyConnected<Dual>>({
    make_pair(&input, &input_gradient),
    make_pair(&bias, &bias_gradient),
    make_pair(&weight, &weight_gradient),
  }, [] () {
    return new Tensor<Dual>({4, 3, 3});
  }, [&bias, &weight] () {
    return new FullyConnected<Dual>(bias, weight);
  }, input, target);

  EXPECT_EQ(Tensor<>({4, 2, 4}, {
    93, 120, 147, 174,   120, 147, 174, 201,
    120, 156, 192, 228,  156, 192, 228, 264,
    147, 192, 237, 282,  192, 237, 282, 327,
    174, 228, 282, 336,  228, 282, 336, 390,
  }), input_gradient);
  EXPECT_EQ(Tensor<>({3, 3}, {
    10, 14, 18,
    14, 18, 22,
    18, 22, 26,
  }), bias_gradient);
  EXPECT_EQ(Tensor<>({3, 3, 2, 4}, {
    30, 40, 50, 60,    40, 50, 60, 70,
    40, 54, 68, 82,    54, 68, 82, 96,
    50, 68, 86, 104,   68, 86, 104, 122,

    40, 54, 68, 82,    54, 68, 82, 96,
    50, 68, 86, 104,   68, 86, 104, 122,
    60, 82, 104, 126,  82, 104, 126, 148,

    50, 68, 86, 104,   68, 86, 104, 122,
    60, 82, 104, 126,  82, 104, 126, 148,
    70, 96, 122, 148,  96, 122, 148, 174
  }), weight_gradient);
}

}  // namespace sacred
