#include <gtest/gtest.h>

#include "dual.hpp"

namespace sacred {

TEST(Dual, Exp) {
  auto x = Dual(1.1, 1);
  auto y = 4.0 * exp(3.0 * x * x);

  EXPECT_FLOAT_EQ(150.85132, y.real);
  EXPECT_FLOAT_EQ(995.61877, y.dual);
}

}  // namespace sacred
