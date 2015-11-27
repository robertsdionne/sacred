#include <gtest/gtest.h>

#include "dual.hpp"
#include "math.hpp"
#include "tensor.hpp"

namespace sacred {

auto math = Math<>();

TEST(Math, GeneralMatrixMultiplication) {
  auto a = Tensor<>({4, 2}, {
    1, 2,
    3, 4,
    5, 6,
    7, 8
  });
  auto b = Tensor<>({2, 3}, {
    1, 2, 3,
    4, 5, 6
  });
  auto c = Tensor<>({4, 3});

  math.GeneralMatrixMultiplication(c, a, b, 0.0, 1.0);

  EXPECT_EQ(
      Tensor<>({4, 3}, {
        9, 12, 15,
        19, 26, 33,
        29, 40, 51,
        39, 54, 69
      }),
      c);
}

}  // namespace sacred
