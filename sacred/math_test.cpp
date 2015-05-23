#include <gtest/gtest.h>

#include "array.hpp"
#include "math.hpp"

using sacred::Array;
using sacred::Math;

auto math = Math<float>();

TEST(MathTest, Convolve2) {
  auto a = Array<float>({3, 3}, {
    9, 8, 7,
    6, 5, 4,
    3, 2, 1
  });
  auto b = Array<float>({4, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  });
  auto c = Array<float>({2, 2}, {
    0, 0,
    0, 0
  });

  math.Convolve2(c, a, b, 0.0, 1.0);

  EXPECT_EQ(
      Array<float>({2, 2}, {
        348, 393,
        528, 573
      }),
      c);
}

TEST(MathTest, GeneralMatrixMultiplication) {
  auto a = Array<float>({4, 2}, {
    1, 2,
    3, 4,
    5, 6,
    7, 8
  });
  auto b = Array<float>({2, 3}, {
    1, 2, 3,
    4, 5, 6
  });
  auto c = Array<float>({4, 3}, {
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
  });

  math.GeneralMatrixMultiplication(c, a, b, 0.0, 1.0);

  EXPECT_EQ(
      Array<float>({4, 3}, {
        9, 12, 15,
        19, 26, 33,
        29, 40, 51,
        39, 54, 69
      }),
      c);
}

TEST(MathTest, RecurrentConvolve2) {
  auto a = Array<float>({3, 3}, {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  });
  auto c = Array<float>({4, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  });

  math.RecurrentConvolve2(c, a, 1.0, 1.0);

  EXPECT_EQ(
      Array<float>({4, 4}, {
        1, 53, 1446, 34685,
        5, 120, 2749, 48191,
        9, 196, 2777, 39825,
        13, 119, 1400, 17795
      }),
      c);
}
