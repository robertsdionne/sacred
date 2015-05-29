#include <gtest/gtest.h>

#include "array.hpp"
#include "dual.hpp"
#include "math.hpp"

using namespace sacred;

TEST(Reconv, Forward) {
  auto output = Array<float>({4, 1}, {
    1, 2, 3, 4
  });
  auto filter = Array<float>({3, 1}, {
    1, 2, 3
  });
  auto math = Math<float>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<float>({4, 1}, {
        1, 3, 8, 21
      }),
      output);
}

TEST(Reconv, BackwardFilter0) {
  auto output = Array<Dual>({4, 1}, {
    1, 2, 3, 4
  });
  auto filter = Array<Dual>({3, 1}, {
    1 + 1_ɛ, 2, 3
  });
  auto math = Math<Dual>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<Dual>({4, 1}, {
        1, 3 + 1_ɛ, 8 + 4_ɛ, 21 + 14_ɛ
      }),
      output);

  auto target = Array<Dual>({4, 1}, {
    1, 2, 7, 20
  });

  auto error = 0_ɛ;
  for (auto i = 0; i < 4; ++i) {
    auto delta = target.at({i}) - output.at({i});
    error += delta * delta / 2.0;
  }

  EXPECT_EQ(1.5 + 19_ɛ, error);
}

TEST(Reconv, BackwardFilter1) {
  auto output = Array<Dual>({4, 1}, {
    1, 2, 3, 4
  });
  auto filter = Array<Dual>({3, 1}, {
    1, 2 + 1_ɛ, 3
  });
  auto math = Math<Dual>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<Dual>({4, 1}, {
        1, 3, 8 + 1_ɛ, 21 + 4_ɛ
      }),
      output);

  auto target = Array<Dual>({4, 1}, {
    1, 2, 7, 20
  });

  auto error = 0_ɛ;
  for (auto i = 0; i < 4; ++i) {
    auto delta = target.at({i}) - output.at({i});
    error += delta * delta / 2.0;
  }

  EXPECT_EQ(1.5 + 5_ɛ, error);
}

TEST(Reconv, BackwardFilter2) {
  auto output = Array<Dual>({4, 1}, {
    1, 2, 3, 4
  });
  auto filter = Array<Dual>({3, 1}, {
    1, 2, 3 + 1_ɛ
  });
  auto math = Math<Dual>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<Dual>({4, 1}, {
        1, 3, 8, 21 + 1_ɛ
      }),
      output);

  auto target = Array<Dual>({4, 1}, {
    1, 2, 7, 20
  });

  auto error = 0_ɛ;
  for (auto i = 0; i < 4; ++i) {
    auto delta = target.at({i}) - output.at({i});
    error += delta * delta / 2.0;
  }

  EXPECT_EQ(1.5 + 1_ɛ, error);
}

TEST(Reconv, Backward) {
  auto output = Array<float>({4, 1}, {
    1, 2, 3, 4
  });
  auto output_diff = Array<float>({4, 1}, {
    0, 1, 1, 1
  });
  auto filter = Array<float>({3, 1}, {
    1, 2, 3
  });
  auto filter_diff = Array<float>({3, 1});
  auto math = Math<float>();

  math.BackwardReconv(filter_diff, filter, output_diff, output);

  EXPECT_EQ(Array<float>({3, 1}, {
        19, 5, 1
      }), filter_diff);
}
