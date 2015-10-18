#include <gtest/gtest.h>
#include <vector>

#include "strides.hpp"

namespace sacred {

  using std::vector;

  TEST(Strides, CStyle) {
    EXPECT_EQ(vector<int>({30, 6, 1}), strides::CStyle({4, 5, 6}));
  }

}  // namespace sacred
