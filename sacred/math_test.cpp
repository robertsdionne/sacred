#include <gtest/gtest.h>

#include "array.hpp"
#include "dual.hpp"
#include "math.hpp"

namespace sacred {

auto math = Math<float>();

TEST(Math, NarrowConvolve2) {
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
  auto c = Array<float>({2, 2});

  math.NarrowConvolve2(c, a, b, 0.0, 1.0);

  EXPECT_EQ(
      Array<float>({2, 2}, {
        348, 393,
        528, 573
      }),
      c);
}

// TEST(Math, BackwardNarrowConvolve2) {
//   auto a = Array<float>({3, 3}, {
//     1, 2, 3,
//     4, 5, 6,
//     7, 8, 9
//   });
//   auto b = Array<float>({4, 4}, {
//     1, 2, 3, 4,
//     5, 6, 7, 8,
//     9, 10, 11, 12,
//     13, 14, 15, 16
//   });
//   auto c = Array<float>({2, 2});
//
//   math.BackwardNarrowConvolve2(c, a, b, 0.0, 1.0);
//
//   EXPECT_EQ(
//       Array<float>({2, 2}, {
//         348, 393,
//         528, 573
//       }),
//       c);
// }

TEST(Math, WideConvolve2) {
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
  auto c = Array<float>({6, 6});

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

TEST(Math, BackwardWideConvolve2) {
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
  auto c = Array<float>({6, 6});

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

TEST(Math, RecurrentConvolve2) {
  auto a = Array<Dual>({3, 3}, {
    9 + 1_ɛ, 8, 7,
    6, 5, 4,
    3, 2, 1
  });
  auto c = Array<Dual>({4, 4}, {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  });
  auto m = Math<Dual>();

  m.RecurrentConvolve2(c, a, 1.0, 1.0);

  EXPECT_EQ(
      Array<Dual>({4, 4}, {
        1, 53 + 5_ɛ, 1446 + 231_ɛ, 34685 + 7670_ɛ,
        5, 120 + 9_ɛ, 2749 + 382_ɛ, 48191 + 7937_ɛ,
        9, 196 + 13_ɛ, 2777 + 224_ɛ, 39825 + 4324_ɛ,
        13, 119, 1400 + 39_ɛ, 17795 + 932_ɛ
      }),
      c);
}

TEST(Math, BackwardRecurrentConvolve2) {
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

TEST(Math, GeneralMatrixMultiplication) {
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
  auto c = Array<float>({4, 3});

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

}  // namespace sacred
