#include <gtest/gtest.h>

#include "convolution.hpp"
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

}  // namespace sacred
