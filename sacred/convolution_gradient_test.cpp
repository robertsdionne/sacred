#include <gtest/gtest.h>
#include <utility>

#include "convolution.hpp"
#include "convolution_gradient.hpp"
#include "dual.hpp"
#include "gradients.hpp"
#include "tensor.hpp"
#include "tensor_testing.hpp"

namespace sacred {

TEST(ConvolutionGradient, Run) {
  auto input = MakeTestTensor<>({4, 4, 4});
  auto input_gradient = Tensor<>({4, 4, 4});
  auto bias_gradient = Tensor<>({3}, {0, 0, 0});
  auto filter = MakeTestTensor<>({3, 3, 3, 4});
  auto filter_gradient = Tensor<>({3, 3, 3, 4});
  auto op = ConvolutionGradient<>(bias_gradient, filter, filter_gradient);
  auto output_gradient = Tensor<>({2, 2, 3}, {
    1, 2, 3,  4, 5, 6,
    2, 3, 4,  5, 6, 7,
  });

  op(output_gradient, input, input_gradient);

  EXPECT_EQ(Tensor<>({4, 4, 4}, {
    14, 20, 26, 32,   52, 73, 94, 115,     73, 94, 115, 136,    62, 77, 92, 107,
    40, 55, 70, 85,   140, 188, 236, 284,  188, 236, 284, 332,  151, 184, 217, 250,
    55, 70, 85, 100,  188, 236, 284, 332,  236, 284, 332, 380,  184, 217, 250, 283,
    38, 47, 56, 65,   121, 148, 175, 202,  148, 175, 202, 229,  110, 128, 146, 164,
  }), input_gradient);
  EXPECT_EQ(Tensor<>({3}, {12, 16, 20}), bias_gradient);
  EXPECT_EQ(Tensor<>({3, 3, 3, 4}, {
    28, 40, 52, 64,
    36, 52, 68, 84,
    44, 64, 84, 104,

    40, 52, 64, 76,
    52, 68, 84, 100,
    64, 84, 104, 124,

    52, 64, 76, 88,
    68, 84, 100, 116,
    84, 104, 124, 144,

    40, 52, 64, 76,
    52, 68, 84, 100,
    64, 84, 104, 124,

    52, 64, 76, 88,
    68, 84, 100, 116,
    84, 104, 124, 144,

    64, 76, 88, 100,
    84, 100, 116, 132,
    104, 124, 144, 164,

    52, 64, 76, 88,
    68, 84, 100, 116,
    84, 104, 124, 144,

    64, 76, 88, 100,
    84, 100, 116, 132,
    104, 124, 144, 164,

    76, 88, 100, 112,
    100, 116, 132, 148,
    124, 144, 164, 184,
  }), filter_gradient);
}

TEST(ConvolutionGradient, Dual) {
  using std::make_pair;

  auto input = MakeTestTensor<Dual>({4, 4, 4});
  auto bias = MakeTestTensor<Dual>({3});
  auto filter = MakeTestTensor<Dual>({3, 3, 3, 4});
  auto target = Tensor<Dual>({2, 2, 3}, {
    822, 984, 1146,   981, 1179, 1377,
    983, 1181, 1379,  1142, 1376, 1610,
  });

  auto input_gradient = Tensor<>({4, 4, 4});
  auto bias_gradient = Tensor<>({3}, {0, 0, 0});
  auto filter_gradient = Tensor<>({3, 3, 3, 4});

  TestGradients<Convolution<Dual>>({
    make_pair(&input, &input_gradient),
    make_pair(&bias, &bias_gradient),
    make_pair(&filter, &filter_gradient),
  }, [] () {
    return new Tensor<Dual>({2, 2, 3});
  }, [&bias, &filter] () {
    return new Convolution<Dual>(bias, filter);
  }, input, target);

  EXPECT_EQ(Tensor<>({4, 4, 4}, {
    14, 20, 26, 32,   52, 73, 94, 115,     73, 94, 115, 136,    62, 77, 92, 107,
    40, 55, 70, 85,   140, 188, 236, 284,  188, 236, 284, 332,  151, 184, 217, 250,
    55, 70, 85, 100,  188, 236, 284, 332,  236, 284, 332, 380,  184, 217, 250, 283,
    38, 47, 56, 65,   121, 148, 175, 202,  148, 175, 202, 229,  110, 128, 146, 164,
  }), input_gradient);
  EXPECT_EQ(Tensor<>({3}, {12, 16, 20}), bias_gradient);
  EXPECT_EQ(Tensor<>({3, 3, 3, 4}, {
    28, 40, 52, 64,
    36, 52, 68, 84,
    44, 64, 84, 104,

    40, 52, 64, 76,
    52, 68, 84, 100,
    64, 84, 104, 124,

    52, 64, 76, 88,
    68, 84, 100, 116,
    84, 104, 124, 144,

    40, 52, 64, 76,
    52, 68, 84, 100,
    64, 84, 104, 124,

    52, 64, 76, 88,
    68, 84, 100, 116,
    84, 104, 124, 144,

    64, 76, 88, 100,
    84, 100, 116, 132,
    104, 124, 144, 164,

    52, 64, 76, 88,
    68, 84, 100, 116,
    84, 104, 124, 144,

    64, 76, 88, 100,
    84, 100, 116, 132,
    104, 124, 144, 164,

    76, 88, 100, 112,
    100, 116, 132, 148,
    124, 144, 164, 184,
  }), filter_gradient);
}

TEST(ConvolutionGradient, DualWithStride) {
  using std::make_pair;

  auto input = MakeTestTensor<Dual>({4, 4, 4});
  auto bias = MakeTestTensor<Dual>({3});
  auto filter = MakeTestTensor<Dual>({3, 3, 3, 4});
  auto target = Tensor<Dual>({2, 2, 3}, {
    822, 984, 1146,   0, 0, 0,
    0, 0, 0,  0, 0, 0,
  });

  auto input_gradient = Tensor<>({4, 4, 4});
  auto bias_gradient = Tensor<>({3}, {0, 0, 0});
  auto filter_gradient = Tensor<>({3, 3, 3, 4});

  TestGradients<Convolution<Dual>>({
    make_pair(&input, &input_gradient),
    make_pair(&bias, &bias_gradient),
    make_pair(&filter, &filter_gradient),
  }, [] () {
    return new Tensor<Dual>({2, 2, 3});
  }, [&bias, &filter] () {
    return new Convolution<Dual>(bias, filter, {2, 2});
  }, input, target);

  EXPECT_EQ(Tensor<>({4, 4, 4}, {
    14, 20, 26, 32,  20, 26, 32, 38,  26, 32, 38, 44,  0, 0, 0, 0,
    20, 26, 32, 38,  26, 32, 38, 44,  32, 38, 44, 50,  0, 0, 0, 0,
    26, 32, 38, 44,  32, 38, 44, 50,  38, 44, 50, 56,  0, 0, 0, 0,
    0, 0, 0, 0,      0, 0, 0, 0,      0, 0, 0, 0,      0, 0, 0, 0,
  }), input_gradient);
  EXPECT_EQ(Tensor<>({3}, {1, 2, 3}), bias_gradient);
  EXPECT_EQ(Tensor<>({3, 3, 3, 4}, {
    1, 2, 3, 4,
    2, 4, 6, 8,
    3, 6, 9, 12,

    2, 3, 4, 5,
    4, 6, 8, 10,
    6, 9, 12, 15,

    3, 4, 5, 6,
    6, 8, 10, 12,
    9, 12, 15, 18,

    2, 3, 4, 5,
    4, 6, 8, 10,
    6, 9, 12, 15,

    3, 4, 5, 6,
    6, 8, 10, 12,
    9, 12, 15, 18,

    4, 5, 6, 7,
    8, 10, 12, 14,
    12, 15, 18, 21,

    3, 4, 5, 6,
    6, 8, 10, 12,
    9, 12, 15, 18,

    4, 5, 6, 7,
    8, 10, 12, 14,
    12, 15, 18, 21,

    5, 6, 7, 8,
    10, 12, 14, 16,
    15, 18, 21, 24,
  }), filter_gradient);
}

}  // namespace sacred
