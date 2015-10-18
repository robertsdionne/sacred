#include <gtest/gtest.h>

#include "checks.hpp"

namespace sacred {

TEST(Checks, CheckState) {
  EXPECT_DEATH(CHECK_STATE(false), "ERROR: false is violated on line 8 of file sacred/checks_test.cpp");
  CHECK_STATE(true);
}

}  // namespace sacred
