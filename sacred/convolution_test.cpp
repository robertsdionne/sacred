#include <gtest/gtest.h>

#include "convolution.hpp"
#include "tensor.hpp"
#include "tensor_testing.hpp"

namespace sacred {

TEST(Convolution, Run) {
  auto input = MakeTestTensor<>({4, 4, 4});
  auto bias = MakeTestTensor<>({3});
  auto filter = MakeTestTensor<>({3, 3, 3, 4});

  auto op = Convolution<>(bias, filter);
  auto output = Tensor<>({2, 2, 3});

  op(input, output);

  EXPECT_EQ(Tensor<>({2, 2, 3}, {
    823, 986, 1149,   985, 1184, 1383,
    985, 1184, 1383,  1147, 1382, 1617,
  }), output);
}

TEST(Convolution, RunWithStride) {
  auto input = MakeTestTensor<>({4, 4, 4});
  auto bias = MakeTestTensor<>({3});
  auto filter = MakeTestTensor<>({3, 3, 3, 4});

  auto op = Convolution<>(bias, filter, {2, 2});
  auto output = Tensor<>({2, 2, 3});

  op(input, output);

  EXPECT_EQ(Tensor<>({2, 2, 3}, {
    823, 986, 1149,  0, 0, 0,
    0, 0, 0,  0, 0, 0,
  }), output);
}

}  // namespace sacred
