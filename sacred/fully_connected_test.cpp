#include <gtest/gtest.h>

#include "fully_connected.hpp"
#include "tensor.hpp"
#include "tensor_testing.hpp"

namespace sacred {

TEST(FullyConnected, RunSimple) {
  auto input = MakeTestTensor<>({1, 1, 4});
  auto bias = MakeTestTensor<>({1, 3});
  auto weight = MakeTestTensor<>({1, 3, 1, 4});//, {
  auto op = FullyConnected<>(bias, weight);
  auto output = Tensor<>({1, 1, 3});

  op(input, output);

  EXPECT_EQ(Tensor<>({1, 1, 3}, {31, 42, 53}), output);
}

TEST(FullyConnected, Run) {
  auto input = MakeTestTensor<>({4, 2, 4});
  auto bias = MakeTestTensor<>({3, 3});
  auto weight = MakeTestTensor<>({3, 3, 2, 4});
  auto op = FullyConnected<>(bias, weight);
  auto output = Tensor<>({4, 3, 3});

  op(input, output);

  EXPECT_EQ(Tensor<>({4, 3, 3}, {
    85, 110, 135,
    110, 135, 160,
    135, 160, 185,

    109, 142, 175,
    142, 175, 208,
    175, 208, 241,

    133, 174, 215,
    174, 215, 256,
    215, 256, 297,

    157, 206, 255,
    206, 255, 304,
    255, 304, 353,
  }), output);
}

}  // namespace sacred
