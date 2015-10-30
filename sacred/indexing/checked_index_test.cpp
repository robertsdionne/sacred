#include <gtest/gtest.h>

#include "checked_index.hpp"

namespace sacred { namespace indexing {

TEST(CheckedIndex, Transform) {
  auto checked_index = CheckedIndex<>();

  EXPECT_DEATH(checked_index.Transform({3}, {1}, {-2}).at(0), "index\\.at\\(i\\)");
  EXPECT_DEATH(checked_index.Transform({3}, {1}, {-1}).at(0), "index\\.at\\(i\\)");
  EXPECT_EQ(0, checked_index.Transform({3}, {1}, {0}).at(0));
  EXPECT_EQ(1, checked_index.Transform({3}, {1}, {1}).at(0));
  EXPECT_EQ(2, checked_index.Transform({3}, {1}, {2}).at(0));
  EXPECT_DEATH(checked_index.Transform({3}, {1}, {3}).at(0), "index\\.at\\(i\\)");
  EXPECT_DEATH(checked_index.Transform({3}, {1}, {4}).at(0), "index\\.at\\(i\\)");
}

}}  // namespaces
