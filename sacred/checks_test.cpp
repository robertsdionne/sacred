#include <gtest/gtest.h>

#include "checks.hpp"

TEST(Checks, CheckState) {
  EXPECT_DEATH(CHECK_STATE(false), "ERROR: false is violated on line 6 of file sacred/checks_test.cpp");
  CHECK_STATE(true);
}
