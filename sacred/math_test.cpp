#include <gtest/gtest.h>

#include "array.hpp"
#include "math.hpp"

using sacred::Array;
using sacred::Math;

auto math = Math<float>();

TEST(MathTest, NarrowConvolve2) {
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

  math.NarrowConvolve2(c, a, b, 0.0, 1.0);

  EXPECT_EQ(
      Array<float>({2, 2}, {
        348, 393,
        528, 573
      }),
      c);
}

TEST(MathTest, BackwardNarrowConvolve2) {
  auto a = Array<float>({3, 3}, {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
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

  math.BackwardNarrowConvolve2(c, a, b, 0.0, 1.0);

  EXPECT_EQ(
      Array<float>({2, 2}, {
        348, 393,
        528, 573
      }),
      c);
}

TEST(MathTest, WideConvolve2) {
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
  auto c = Array<float>({6, 6}, {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
  });

  math.WideConvolve2(c, a, b, 0.0, 1.0);

  EXPECT_EQ(
      Array<float>({6, 6}, {
        9, 26, 50, 74, 53, 28,
        51, 111, 178, 217, 145, 72,
        114, 231, 348, 393, 252, 120,
        186, 363, 528, 573, 360, 168,
        105, 197, 274, 295, 175, 76,
        39, 68, 86, 92, 47, 16
      }),
      c);
}

TEST(MathTest, BackwardWideConvolve2) {
  auto a = Array<float>({3, 3}, {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  });
  auto b = Array<float>({4, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  });
  auto c = Array<float>({6, 6}, {
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
  });

  math.BackwardWideConvolve2(c, a, b, 0.0, 1.0);

  EXPECT_EQ(
      Array<float>({6, 6}, {
        9, 26, 50, 74, 53, 28,
        51, 111, 178, 217, 145, 72,
        114, 231, 348, 393, 252, 120,
        186, 363, 528, 573, 360, 168,
        105, 197, 274, 295, 175, 76,
        39, 68, 86, 92, 47, 16
      }),
      c);
}

TEST(MathTest, RecurrentConvolve2) {
  auto a = Array<float>({3, 3}, {
    9, 8, 7,
    6, 5, 4,
    3, 2, 1
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

TEST(MathTest, BackwardRecurrentConvolve2) {
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

  math.BackwardRecurrentConvolve2(c, a, 1.0, 1.0);

  EXPECT_EQ(
      Array<float>({4, 4}, {
        20490, 1275, 75, 4,
        23467, 1986, 127, 8,
        15515, 1694, 179, 12,
        5296, 661, 91, 16
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
