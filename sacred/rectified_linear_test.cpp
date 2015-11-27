#include <gtest/gtest.h>

#include "rectified_linear.hpp"
#include "tensor.hpp"

namespace sacred {

TEST(Rectifier, Run) {
  auto input = Tensor<>({3, 4}, {
    1, -2, 3, -4,
    -5, 6, -7, 8,
    9, -10, 11, -12,
  });
  auto op = RectifiedLinear<>();
  auto output = Tensor<>({3, 4}, {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  });

  op(input, output);

  EXPECT_EQ(Tensor<>({3, 4}, {
    1, 0, 3, 0,
    0, 6, 0, 8,
    9, 0, 11, 0,
  }), output);
}

}  // namespace sacred
