#include <gtest/gtest.h>

#include "wrapped_index.hpp"

namespace sacred {

TEST(WrappedIndex, Transform) {
  auto wrapped_index = WrappedIndex<>();

  EXPECT_EQ(2, wrapped_index.Transform({3}, {1}, {-1}).at(0));
  EXPECT_EQ(0, wrapped_index.Transform({3}, {1}, {0}).at(0));
  EXPECT_EQ(1, wrapped_index.Transform({3}, {1}, {1}).at(0));
  EXPECT_EQ(2, wrapped_index.Transform({3}, {1}, {2}).at(0));
  EXPECT_EQ(0, wrapped_index.Transform({3}, {1}, {3}).at(0));
}

}  // namespace sacred
