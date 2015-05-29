#include <gtest/gtest.h>

#include "array.hpp"
#include "dual.hpp"
#include "math.hpp"

using namespace sacred;

TEST(Reconv, Forward) {
  auto output = Array<float>({8, 1}, {
    1, 2, 3, 4, 5, 6, 7, 8
  });
  auto filter = Array<float>({4, 1}, {
    1, 2, 3, 4
  });
  auto math = Math<float>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<float>({8, 1}, {
        1, 3, 8, 21, 55, 139, 351, 886
      }),
      output);
}

TEST(Reconv, BackwardFilter0) {
  auto output = Array<Dual>({8, 1}, {
    1, 2, 3, 4, 5, 6, 7, 8
  });
  auto filter = Array<Dual>({4, 1}, {
    1 + 1_ɛ, 2, 3, 4
  });
  auto math = Math<Dual>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<Dual>({8, 1}, {
        1, 3 + 1_ɛ, 8 + 4_ɛ, 21 + 14_ɛ, 55 + 46_ɛ, 139 + 145_ɛ, 351 + 434_ɛ, 886 + 1269_ɛ
      }),
      output);

  auto target = Array<Dual>({8, 1}, {
    1, 2, 7, 20, 54, 138, 350, 885
  });

  auto error = 0_ɛ;
  for (auto i = 0; i < 8; ++i) {
    auto delta = target.at({i}) - output.at({i});
    error += delta * delta / 2.0;
  }

  EXPECT_EQ(3.5 + 1913_ɛ, error);
}

TEST(Reconv, BackwardFilter1) {
  auto output = Array<Dual>({8, 1}, {
    1, 2, 3, 4, 5, 6, 7, 8
  });
  auto filter = Array<Dual>({4, 1}, {
    1, 2 + 1_ɛ, 3, 4
  });
  auto math = Math<Dual>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<Dual>({8, 1}, {
        1, 3, 8 + 1_ɛ, 21 + 4_ɛ, 55 + 14_ɛ, 139 + 46_ɛ, 351 + 145_ɛ, 886 + 434_ɛ
      }),
      output);

  auto target = Array<Dual>({8, 1}, {
    1, 2, 7, 20, 54, 138, 350, 885
  });

  auto error = 0_ɛ;
  for (auto i = 0; i < 8; ++i) {
    auto delta = target.at({i}) - output.at({i});
    error += delta * delta / 2.0;
  }

  EXPECT_EQ(3.5 + 644_ɛ, error);
}

TEST(Reconv, BackwardFilter2) {
  auto output = Array<Dual>({8, 1}, {
    1, 2, 3, 4, 5, 6, 7, 8
  });
  auto filter = Array<Dual>({4, 1}, {
    1, 2, 3 + 1_ɛ, 4
  });
  auto math = Math<Dual>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<Dual>({8, 1}, {
        1, 3, 8, 21 + 1_ɛ, 55 + 4_ɛ, 139 + 14_ɛ, 351 + 46_ɛ, 886 + 145_ɛ
      }),
      output);

  auto target = Array<Dual>({8, 1}, {
    1, 2, 7, 20, 54, 138, 350, 885
  });

  auto error = 0_ɛ;
  for (auto i = 0; i < 8; ++i) {
    auto delta = target.at({i}) - output.at({i});
    error += delta * delta / 2.0;
  }

  EXPECT_EQ(3.5 + 210_ɛ, error);
}

TEST(Reconv, BackwardFilter3) {
  auto output = Array<Dual>({8, 1}, {
    1, 2, 3, 4, 5, 6, 7, 8
  });
  auto filter = Array<Dual>({4, 1}, {
    1, 2, 3, 4 + 1_ɛ
  });
  auto math = Math<Dual>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<Dual>({8, 1}, {
        1, 3, 8, 21, 55 + 1_ɛ, 139 + 4_ɛ, 351 + 14_ɛ, 886 + 46_ɛ
      }),
      output);

  auto target = Array<Dual>({8, 1}, {
    1, 2, 7, 20, 54, 138, 350, 885
  });

  auto error = 0_ɛ;
  for (auto i = 0; i < 8; ++i) {
    auto delta = target.at({i}) - output.at({i});
    error += delta * delta / 2.0;
  }

  EXPECT_EQ(3.5 + 65_ɛ, error);
}

TEST(Reconv, Backward) {
  auto output = Array<float>({8, 1}, {
    1, 2, 3, 4, 5, 6, 7, 8
  });
  auto output_diff = Array<float>({8, 1}, {
    0, 1, 1, 1, 1, 1, 1, 1
  });
  auto filter = Array<float>({4, 1}, {
    1, 2, 3, 4
  });
  auto filter_diff = Array<float>({4, 1});
  auto math = Math<float>();

  math.BackwardReconv(filter_diff, filter, output_diff, output);

  EXPECT_EQ(Array<float>({4, 1}, {
        1913, 644, 210, 65
      }), filter_diff);
}
