#include <gtest/gtest.h>

#include "array.hpp"
#include "dual.hpp"
#include "math.hpp"

using namespace sacred;

TEST(Reconv, Forward) {
  auto output = Array<float>({8}, {
    1, 2, 3, 4, 5, 6, 7, 8
  });
  auto filter = Array<float>({4}, {
    1, 2, 3, 4
  });
  auto math = Math<float>();

  math.Reconv(output, filter);

  EXPECT_EQ(Array<float>({8}, {
        1, 3, 8, 21, 55, 139, 351, 886
      }),
      output);
}

TEST(Reconv, BackwardFilter) {
  auto math = Math<Dual>();
  auto target = Array<Dual>({8}, {
    1, 2, 6, 20, 54, 138, 350, 885
  });
  auto error = Array<Dual>({4});

  for (auto k = 0; k < 4; ++k) {
    auto output = Array<Dual>({8}, {
      1, 2, 3, 4, 5, 6, 7, 8
    });
    auto filter = Array<Dual>({4}, {
      1, 2, 3, 4
    });
    filter.add({k}, 1_ɛ);

    math.Reconv(output, filter);

    for (auto i = 0; i < 8; ++i) {
      auto delta = target.at({i}) - output.at({i});
      error.add({k}, delta * delta / 2.0);
    }
  }

  EXPECT_EQ(Array<Dual>({4}, {
    5.0 + 1917_ɛ, 5.0 + 645_ɛ, 5.0 + 210_ɛ, 5.0 + 65_ɛ
  }), error);
}

TEST(Reconv, Backward) {
  auto output = Array<float>({8}, {
    1, 3, 8, 21, 55, 139, 351, 886
  });
  auto output_diff = Array<float>({8}, {
    0, 1, 2, 1, 1, 1, 1, 1
  });
  auto filter = Array<float>({4}, {
    1, 2, 3, 4
  });
  auto filter_diff = Array<float>({4});
  auto math = Math<float>();

  math.BackwardReconv(filter_diff, filter, output_diff, output);

  EXPECT_EQ(Array<float>({4}, {
        1917, 645, 210, 65
      }), filter_diff);
}
