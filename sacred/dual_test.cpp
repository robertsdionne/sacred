#include <algorithm>
#include <gtest/gtest.h>

#include "dual.hpp"

namespace sacred {

TEST(Dual, Exp) {
  auto x = Dual(1.1, 1);
  auto y = 4.0 * exp(3.0 * x * x);

  EXPECT_FLOAT_EQ(150.85132, y.real);
  EXPECT_FLOAT_EQ(995.61877, y.dual);
}

TEST(Dual, LessThan) {
  auto x = Dual(1), y = Dual(2);

  EXPECT_TRUE(x < y);
  EXPECT_FALSE(y < x);
}

TEST(Dual, Minimum) {
  EXPECT_FLOAT_EQ(0, std::max(Dual(0), Dual(-1, 1)).real);
  EXPECT_FLOAT_EQ(1, std::max(Dual(0), Dual(1, 1)).real);

  EXPECT_FLOAT_EQ(0, std::max(Dual(0), Dual(-1, 1)).dual);
  EXPECT_FLOAT_EQ(1, std::max(Dual(0), Dual(1, 1)).dual);
}

}  // namespace sacred
