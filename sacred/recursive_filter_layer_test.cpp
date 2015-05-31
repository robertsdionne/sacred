#include <gtest/gtest.h>
#include <iostream>

#include "array.hpp"
#include "blob.hpp"
#include "dual.hpp"
#include "recursive_filter_layer.hpp"

using namespace sacred;

TEST(RecursiveFilterLayer, Forward) {
  auto input = Blob<float>({4, 4}, {
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7
  });
  auto bias = Blob<float>({4});
  auto filter = Blob<float>({3, 3}, {
    1, 2, 3,
    2, 3, 4,
    4, 5, 6
  });
  auto layer = RecursiveFilterLayer<float>(bias, filter);
  auto output = Blob<float>({4, 4});
  layer.Forward(input, &output);
  EXPECT_EQ(
      Array<float>({4, 4}, {
        1, 6, 36, 227,
        2, 14, 95, 635,
        3, 22, 157, 1093,
        4, 25, 171, 1196
      }),
      output.value());
}

TEST(RecursiveFilterLayer, Backward) {
  auto output = Blob<float>({4, 4}, {
    1, 6, 36, 227,
    2, 14, 95, 635,
    3, 22, 157, 1093,
    4, 25, 171, 1196
  }, {
    0, 1, 2, 3,
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6
  });
  auto bias = Blob<float>({4, 1});
  auto filter = Blob<float>({3, 3}, {
    1, 2, 3,
    2, 3, 4,
    4, 5, 6
  });
  auto layer = RecursiveFilterLayer<float>(bias, filter);
  auto input = Blob<float>({4, 4}, {
    1, 2, 3, 4,
    2, 3, 4, 5,
    3, 4, 5, 6,
    4, 5, 6, 7
  });
  layer.Backward(output, &input);
  EXPECT_EQ(Array<float>({4, 1}, {
    6, 10, 14, 18
  }), bias.diff());
  EXPECT_EQ(Array<float>({3, 3}, {
    6293, 573, 38,
    6475, 640, 50,
    4020, 410, 32
  }), filter.diff());
  EXPECT_EQ(Array<float>({4, 4}, {
    1926, 214, 24, 3,
    2288, 305, 34, 4,
    1675, 262, 42, 5,
    690, 118, 22, 6
  }), input.diff());
}

TEST(RecursiveFilterLayer, GradientInput) {
  auto target = Array<Dual>({4, 4}, {
    1, 6-1, 36-2, 227-3,
    2-1, 14-2, 95-3, 635-4,
    3-2, 22-3, 157-4, 1093-5,
    4-3, 25-4, 171-5, 1196-6
  });
  auto error = Array<Dual>({4, 4});

  for (auto k = 0; k < 4; ++k) {
    for (auto l = 0; l < 4; ++l) {
      auto input = Blob<Dual>({4, 4}, {
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7
      });
      input.value({k, l}) += 1_ɛ;
      auto bias = Blob<Dual>({4});
      auto filter = Blob<Dual>({3, 3}, {
        1, 2, 3,
        2, 3, 4,
        4, 5, 6
      });
      auto layer = RecursiveFilterLayer<Dual>(bias, filter);
      auto output = Blob<Dual>({4, 4});
      layer.Forward(input, &output);

      for (auto i = 0; i < 4; ++i) {
        for (auto j = 0; j < 4; ++j) {
          auto delta = target.at({i, j}) - output.value({i, j});
          error.at({k, l}) += delta * delta / 2.0;
        }
      }
    }
  }

  EXPECT_EQ(Array<Dual>({4, 4}, {
    92 + 1926_ɛ, 92 + 214_ɛ, 92 + 24_ɛ, 92 + 3_ɛ,
    92 + 2288_ɛ, 92 + 305_ɛ, 92 + 34_ɛ, 92 + 4_ɛ,
    92 + 1675_ɛ, 92 + 262_ɛ, 92 + 42_ɛ, 92 + 5_ɛ,
    92 + 690_ɛ, 92 + 118_ɛ, 92 + 22_ɛ, 92 + 6_ɛ
  }), error);
}

TEST(RecursiveFilterLayer, GradientFilter) {
  auto target = Array<Dual>({4, 4}, {
    1, 6-1, 36-2, 227-3,
    2-1, 14-2, 95-3, 635-4,
    3-2, 22-3, 157-4, 1093-5,
    4-3, 25-4, 171-5, 1196-6
  });
  auto error = Array<Dual>({3, 3});

  for (auto k = 0; k < 3; ++k) {
    for (auto l = 0; l < 3; ++l) {
      auto input = Blob<Dual>({4, 4}, {
        1, 2, 3, 4,
        2, 3, 4, 5,
        3, 4, 5, 6,
        4, 5, 6, 7
      });
      auto bias = Blob<Dual>({4});
      auto filter = Blob<Dual>({3, 3}, {
        1, 2, 3,
        2, 3, 4,
        4, 5, 6
      });
      filter.value({k, l}) += 1_ɛ;
      auto layer = RecursiveFilterLayer<Dual>(bias, filter);
      auto output = Blob<Dual>({4, 4});
      layer.Forward(input, &output);

      for (auto i = 0; i < 4; ++i) {
        for (auto j = 0; j < 4; ++j) {
          auto delta = target.at({i, j}) - output.value({i, j});
          error.at({k, l}) += delta * delta / 2.0;
        }
      }
    }
  }

  EXPECT_EQ(Array<Dual>({3, 3}, {
    92 + 6293_ɛ, 92 + 573_ɛ, 92 + 38_ɛ,
    92 + 6475_ɛ, 92 + 640_ɛ, 92 + 50_ɛ,
    92 + 4020_ɛ, 92 + 410_ɛ, 92 + 32_ɛ
  }), error);
}

TEST(RecursiveFilterLayer, GradientBias) {
  auto target = Array<Dual>({4, 4}, {
    1, 6-1, 36-2, 227-3,
    2-1, 14-2, 95-3, 635-4,
    3-2, 22-3, 157-4, 1093-5,
    4-3, 25-4, 171-5, 1196-6
  });
  auto error = Array<Dual>({4, 1});

  for (auto k = 0; k < 4; ++k) {
    auto input = Blob<Dual>({4, 4}, {
      1, 2, 3, 4,
      2, 3, 4, 5,
      3, 4, 5, 6,
      4, 5, 6, 7
    });
    auto bias = Blob<Dual>({4});
    bias.value({k}) += 1_ɛ;
    auto filter = Blob<Dual>({3, 3}, {
      1, 2, 3,
      2, 3, 4,
      4, 5, 6
    });
    auto layer = RecursiveFilterLayer<Dual>(bias, filter);
    auto output = Blob<Dual>({4, 4});
    layer.Forward(input, &output);

    for (auto i = 0; i < 4; ++i) {
      for (auto j = 0; j < 4; ++j) {
        auto delta = target.at({i, j}) - output.value({i, j});
        error.at({k}) += delta * delta / 2.0;
      }
    }
  }

  EXPECT_EQ(Array<Dual>({4, 1}, {
    92 + 6_ɛ, 92 + 10_ɛ, 92 + 14_ɛ, 92 + 18_ɛ
  }), error);
}
