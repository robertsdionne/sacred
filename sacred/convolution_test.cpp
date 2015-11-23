#include <gtest/gtest.h>
#include <tuple>
#include <utility>

#include "convolution.hpp"
#include "dual.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(Convolution, Run) {
  auto input = Tensor<>({4, 4, 4}, {
    1, 2, 3, 4,  2, 3, 4, 5,  3, 4, 5, 6,  4, 5, 6, 7,
    2, 3, 4, 5,  3, 4, 5, 6,  4, 5, 6, 7,  5, 6, 7, 8,
    3, 4, 5, 6,  4, 5, 6, 7,  5, 6, 7, 8,  6, 7, 8, 9,
    4, 5, 6, 7,  5, 6, 7, 8,  6, 7, 8, 9,  7, 8, 9, 10,
  });
  auto bias = Tensor<>({3}, {1, 2, 3});
  auto filter = Tensor<>({3, 3, 3, 4}, { // h, w, n, c
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,

    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7,

    3, 4, 5, 6,
    4, 5, 6, 7,
    5, 6, 7, 8,

    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7,

    3, 4, 5, 6,
    4, 5, 6, 7,
    5, 6, 7, 8,

    4, 5, 6, 7,
    5, 6, 7, 8,
    6, 7, 8, 9,

    3, 4, 5, 6,
    4, 5, 6, 7,
    5, 6, 7, 8,

    4, 5, 6, 7,
    5, 6, 7, 8,
    6, 7, 8, 9,

    5, 6, 7, 8,
    6, 7, 8, 9,
    7, 8, 9, 10,
  });

  auto op = Convolution<>(bias, filter);
  auto output = Tensor<>({2, 2, 3});

  op(input, output);

  EXPECT_EQ(Tensor<>({2, 2, 3}, {
    823, 986, 1149,   985, 1184, 1383,
    985, 1184, 1383,  1147, 1382, 1617,
  }), output);
}

TEST(Convolution, Gradient) {
  using std::make_pair;
  using std::tie;

  auto input = Tensor<Dual>({4, 4, 4}, {
    1, 2, 3, 4,  2, 3, 4, 5,  3, 4, 5, 6,  4, 5, 6, 7,
    2, 3, 4, 5,  3, 4, 5, 6,  4, 5, 6, 7,  5, 6, 7, 8,
    3, 4, 5, 6,  4, 5, 6, 7,  5, 6, 7, 8,  6, 7, 8, 9,
    4, 5, 6, 7,  5, 6, 7, 8,  6, 7, 8, 9,  7, 8, 9, 10,
  });
  auto bias = Tensor<Dual>({3}, {1, 2, 3});
  auto filter = Tensor<Dual>({3, 3, 3, 4}, { // h, w, n, c
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,

    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7,

    3, 4, 5, 6,
    4, 5, 6, 7,
    5, 6, 7, 8,

    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7,

    3, 4, 5, 6,
    4, 5, 6, 7,
    5, 6, 7, 8,

    4, 5, 6, 7,
    5, 6, 7, 8,
    6, 7, 8, 9,

    3, 4, 5, 6,
    4, 5, 6, 7,
    5, 6, 7, 8,

    4, 5, 6, 7,
    5, 6, 7, 8,
    6, 7, 8, 9,

    5, 6, 7, 8,
    6, 7, 8, 9,
    7, 8, 9, 10,
  });
  auto target = Tensor<Dual>({2, 2, 3}, {
    822, 984, 1146,   981, 1179, 1377,
    983, 1181, 1379,  1142, 1376, 1610,
  });

  auto input_gradient = Tensor<>({4, 4, 4});
  auto bias_gradient = Tensor<>({3}, {0, 0, 0});
  auto filter_gradient = Tensor<>({3, 3, 3, 4});

  for (auto pair : {
    make_pair(&input, &input_gradient),
    make_pair(&bias, &bias_gradient),
    make_pair(&filter, &filter_gradient),
  }) {
    Tensor<Dual> *parameter;
    Tensor<> *parameter_gradient;
    tie(parameter, parameter_gradient) = pair;

    for (auto i = 0; i < parameter->size(); ++i) {
      auto output = Tensor<Dual>({2, 2, 3});
      auto op = Convolution<Dual>(bias, filter);

      parameter->data(i) += 1_ɛ;

      op(input, output);

      auto loss = 0_ɛ;
      for (auto j = 0; j < target.size(); ++j) {
        auto delta = target.data(j) - output.data(j);
        loss += delta * delta / 2.0f;
      }

      parameter_gradient->data(i) = loss.dual;

      parameter->data(i).dual = 0;
    }
  }

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

}  // namespace sacred